#!/usr/bin/env python

"""
pose_chained.py: Chained predictions
Adapted from: https://github.com/Naman-ntc/Pytorch-Human-Pose-Estimation
"""
__author__ = "David Pascual Hernandez"
__date__ = "2019/11/28"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from pose import PoseEstimator


def get_transform(center, scale, rot, res):
    h = scale
    t = np.eye(3)

    t[0, 0] = res / h
    t[1, 1] = res / h
    t[0, 2] = res * (- center[0] / h + 0.5)
    t[1, 2] = res * (- center[1] / h + 0.5)
    if rot != 0:
        rot = -rot
        r = np.eye(3)
        ang = rot * np.math.pi / 180
        s = np.math.sin(ang)
        c = np.math.cos(ang)
        r[0, 0] = c
        r[0, 1] = - s
        r[1, 0] = s
        r[1, 1] = c
        t_ = np.eye(3)
        t_[0, 2] = - res / 2
        t_[1, 2] = - res / 2
        t_inv = np.eye(3)
        t_inv[0, 2] = res / 2
        t_inv[1, 2] = res / 2
        t = np.dot(np.dot(np.dot(t_inv, r), t_), t)

    return t


def transform(pt, center, scale, rot, res, invert=False):
    pt_ = np.ones(3)
    pt_[0], pt_[1] = pt[0], pt[1]

    t = get_transform(center, scale, rot, res)
    if invert:
        t = np.linalg.inv(t)
    new_point = np.dot(t, pt_)[:2]
    new_point = new_point.astype(np.int32)
    return new_point


def crop_human(img, center, scale, rot, res):
    ht, wd = img.shape[0], img.shape[1]
    tmpImg, newImg = img.copy(), np.zeros((res, res, 3), dtype=np.uint8)

    scaleFactor = scale / res
    if scaleFactor < 2:
        scaleFactor = 1
    else:
        newSize = int(np.math.floor(max(ht, wd) / scaleFactor))
        newSize_ht = int(np.math.floor(ht / scaleFactor))
        newSize_wd = int(np.math.floor(wd / scaleFactor))
        if newSize < 2:
            return torch.from_numpy(newImg.transpose(2, 0, 1).astype(np.float32) / 256.)
        else:
            tmpImg = cv2.resize(tmpImg, (newSize_wd, newSize_ht))  # TODO
            ht, wd = tmpImg.shape[0], tmpImg.shape[1]

    c, s = 1.0 * center / scaleFactor, scale / scaleFactor
    c[0], c[1] = c[1], c[0]
    ul = transform((0, 0), c, s, 0, res, invert=True)
    br = transform((res, res), c, s, 0, res, invert=True)

    if scaleFactor >= 2:
        br = br - (br - ul - res)

    pad = int(np.math.ceil((((ul - br) ** 2).sum() ** 0.5) / 2 - (br[0] - ul[0]) / 2))
    if rot != 0:
        ul = ul - pad
        br = br + pad

    old_ = [max(0, ul[0]), min(br[0], ht), max(0, ul[1]), min(br[1], wd)]
    new_ = [max(0, - ul[0]), min(br[0], ht) - ul[0], max(0, - ul[1]), min(br[1], wd) - ul[1]]

    newImg = np.zeros((br[0] - ul[0], br[1] - ul[1], 3), dtype=np.uint8)
    # print 'new old newshape tmpshape center', new_[0], new_[1], old_[0], old_[1], newImg.shape, tmpImg.shape, center
    try:
        newImg[new_[0]:new_[1], new_[2]:new_[3], :] = tmpImg[old_[0]:old_[1], old_[2]:old_[3], :]
    except:
        # print 'ERROR: new old newshape tmpshape center', new_[0], new_[1], old_[0], old_[1], newImg.shape, tmpImg.shape, center
        return np.zeros((3, res, res), np.uint8)
    if rot != 0:
        M = cv2.getRotationMatrix2D((newImg.shape[0] / 2, newImg.shape[1] / 2), rot, 1)
        newImg = cv2.warpAffine(newImg, M, (newImg.shape[0], newImg.shape[1]))
        newImg = newImg[pad + 1:-pad + 1, pad + 1:-pad + 1, :].copy()

    if scaleFactor < 2:
        newImg = cv2.resize(newImg, (res, res))

    return newImg.transpose(2, 0, 1).astype(np.float32)


def map_resize(new_shape, heatmap):
    # Resizes the output back to the size of the test image
    scale_y = new_shape[0] / float(heatmap.shape[0])
    scale_x = new_shape[1] / float(heatmap.shape[1])
    map_resized = cv2.resize(heatmap, None, fx=scale_x, fy=scale_y,
                             interpolation=cv2.INTER_CUBIC)

    return map_resized


class PoseChained(PoseEstimator):
    def __init__(self, model_fname, boxsize):
        """
        Constructs Estimator class.
        @param model_fname: Pytorch model
        """
        PoseEstimator.__init__(self, model_fname, boxsize)
        self.device = None

    def init_net(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state = torch.load(self.model_fname, map_location=self.device)

        self.net = ChainedPredictions(modelName='resnet34', hhKernel=1, ohKernel=1, nJoints=16)

        self.net = self.net.to(self.device)
        self.net.load_state_dict(state["model_state"])
        self.net.eval()

    def estimate(self):
        """
        Estimates human pose.
        @return: np.array: joints heatmaps
        """
        if not self.net:
            self.init_net()
        output_torch = self.net(self.im)
        return output_torch[-1].cpu().detach().numpy().squeeze()

    def get_coords(self, sample, human_bbox):
        """
        Estimate human pose given an input image.
        @param sample: np.array - original input image
        @param human_bbox: np.array - human coordinates
        @return: np.array - joint coords
        """
        (ux, uy), (lx, ly) = human_bbox

        # get center and scale
        center = np.array((int((ux + lx) / 2), int((uy + ly) / 2)))
        scale = float(max(abs(ux - lx), abs(uy - ly)))

        # transform image
        image_transformed = crop_human(sample, center, scale, rot=0, res=self.boxsize) / 256
        image_transformed = np.array([image_transformed])
        self.im = torch.from_numpy(np.ascontiguousarray(image_transformed)).cuda()
        self.im.to(self.device, non_blocking=True).float()

        # get heatmaps
        output = self.estimate()

        # store joints
        joint_coords = []
        for map in output:
            # plt.figure(), plt.imshow(map_0), plt.show()
            joint = np.array(np.unravel_index(map.argmax(), map.shape))
            resize_factor = scale / map.shape[0]
            joint = joint * resize_factor
            joint = (int(joint[1] + center[0] - scale / 2), int(joint[0] + center[1] - scale / 2))
            joint_coords.append(joint)

        joint_coords_sorted = []
        joint_coords_sorted.append(joint_coords[9])  # head top
        joint_coords_sorted.append(joint_coords[8])  # upper neck
        joint_coords_sorted.append(joint_coords[12])  # shoulder right
        joint_coords_sorted.append(joint_coords[11])  # elbow right
        joint_coords_sorted.append(joint_coords[10])  # wrist right
        joint_coords_sorted.append(joint_coords[13])  # shoulder left
        joint_coords_sorted.append(joint_coords[14])  # elbow left
        joint_coords_sorted.append(joint_coords[15])  # wrist left
        joint_coords_sorted.append(joint_coords[2])  # right hip
        joint_coords_sorted.append(joint_coords[1])  # right knee
        joint_coords_sorted.append(joint_coords[0])  # right ankle
        joint_coords_sorted.append(joint_coords[3])  # left hip
        joint_coords_sorted.append(joint_coords[4])  # left knee
        joint_coords_sorted.append(joint_coords[5])  # left ankle

        return joint_coords_sorted


############################
# PyTorch model definition #
############################

class Identity(nn.Module):
    """docstring for Identity"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Deception(nn.Module):
    """docstring for Deception"""

    def __init__(self, hiddenChans):
        super(Deception, self).__init__()
        self.hiddenChans = hiddenChans

        _stack1 = []
        _stack2 = []
        _stack3 = []

        self.start = nn.Conv2d(self.hiddenChans, 32, 1)

        _stack1.append(nn.ConvTranspose2d(32, 32, 2, 2, 0))
        _stack1.append(nn.ConvTranspose2d(32, 32, 4, 2, 1))
        _stack1.append(nn.ConvTranspose2d(32, 32, 6, 2, 2))
        _stack1.append(nn.BatchNorm2d(32))
        self.stack1 = nn.ModuleList(_stack1)

        _stack2.append(nn.ConvTranspose2d(32, 32, 2, 2, 0))
        _stack2.append(nn.ConvTranspose2d(32, 32, 4, 2, 1))
        _stack2.append(nn.ConvTranspose2d(32, 32, 6, 2, 2))
        _stack2.append(nn.BatchNorm2d(32))
        self.stack2 = nn.ModuleList(_stack2)

        self.end = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, x):
        x = self.start(x)
        x = self.stack1[0](x) + self.stack1[1](x) + self.stack1[2](x)
        x = self.stack2[0](x) + self.stack2[1](x) + self.stack2[2](x)
        x = self.end(x)
        return x


class ChainedPredictions(nn.Module):
    """docstring for ChainedPredictions"""

    def __init__(self, modelName, hhKernel, ohKernel, nJoints):
        super(ChainedPredictions, self).__init__()
        self.nJoints = nJoints
        self.modelName = modelName
        self.resnet = getattr(torchvision.models, self.modelName)(pretrained=True)
        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()
        self.hiddenChans = 64  ### Add cases!

        self.hhKernel = hhKernel
        self.ohKernel = ohKernel

        self.init_hidden = nn.Conv2d(512, self.hiddenChans, 1)
        _deception = []
        for i in range(self.nJoints):
            _deception.append(Deception(self.hiddenChans))
        self.deception = nn.ModuleList(_deception)

        _h2h = []
        _o2h = []
        for i in range(nJoints):
            _o = []
            _h2h.append(
                nn.Sequential(
                    nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=self.hhKernel,
                              padding=self.hhKernel // 2),
                    nn.BatchNorm2d(self.hiddenChans)
                )
            )
            for j in range(i + 1):
                _o.append(nn.Sequential(
                    nn.Conv2d(1, self.hiddenChans, 1),
                    nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=self.ohKernel, stride=2,
                              padding=self.ohKernel // 2),
                    nn.BatchNorm2d(self.hiddenChans),
                    nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=self.ohKernel, stride=2,
                              padding=self.ohKernel // 2),
                    nn.BatchNorm2d(self.hiddenChans),
                )
                )
            _o2h.append(nn.ModuleList(_o))

        self.h2h = nn.ModuleList(_h2h)
        self.o2h = nn.ModuleList(_o2h)

    def forward(self, x):
        hidden = [0] * self.nJoints
        output = [None] * self.nJoints
        hidden[0] += self.resnet(x).reshape(-1, 512, 8, 8)
        hidden[0] = self.init_hidden(hidden[0])
        output[0] = self.deception[0](hidden[0])

        for i in range(self.nJoints - 1):
            hidden[i + 1] = self.h2h[i](hidden[i])
            for j in range(i + 1):
                hidden[i + 1] += self.o2h[i][j](output[j])
            hidden[i + 1] = torch.relu(hidden[i + 1])
            output[i + 1] = self.deception[i + 1](hidden[i + 1])
        return torch.cat(output, 1)