#!/usr/bin/env python

"""
pose_stacked.py: Stacked Hourglass
Adapted from: https://github.com/Naman-ntc/Pytorch-Human-Pose-Estimation
"""
__author__ = "David Pascual Hernandez"
__date__ = "2019/11/28"

import cv2
import numpy as np
import torch
import torch.nn as nn

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


class PoseStacked(PoseEstimator):
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

        self.net = StackedHourGlass(nChannels=256, nStack=2, nModules=2, numReductions=4, nJoints=16)

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

class BnReluConv(nn.Module):
    """docstring for BnReluConv"""

    def __init__(self, inChannels, outChannels, kernelSize=1, stride=1, padding=0):
        super(BnReluConv, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding

        self.bn = nn.BatchNorm2d(self.inChannels)
        self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):
    """docstring for ConvBlock"""

    def __init__(self, inChannels, outChannels):
        super(ConvBlock, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.outChannelsby2 = outChannels // 2

        self.cbr1 = BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0)
        self.cbr2 = BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
        self.cbr3 = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        return x


class SkipLayer(nn.Module):
    """docstring for SkipLayer"""

    def __init__(self, inChannels, outChannels):
        super(SkipLayer, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        if (self.inChannels == self.outChannels):
            self.conv = None
        else:
            self.conv = nn.Conv2d(self.inChannels, self.outChannels, 1)

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        return x


class Residual(nn.Module):
    """docstring for Residual"""

    def __init__(self, inChannels, outChannels):
        super(Residual, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.cb = ConvBlock(inChannels, outChannels)
        self.skip = SkipLayer(inChannels, outChannels)

    def forward(self, x):
        out = 0
        out = out + self.cb(x)
        out = out + self.skip(x)
        return out


class myUpsample(nn.Module):
    def __init__(self):
        super(myUpsample, self).__init__()
        pass

    def forward(self, x):
        return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2) * 2,
                                                                              x.size(3) * 2)


class Hourglass(nn.Module):
    """docstring for Hourglass"""

    def __init__(self, nChannels=256, numReductions=4, nModules=2, poolKernel=(2, 2), poolStride=(2, 2),
                 upSampleKernel=2):
        super(Hourglass, self).__init__()
        self.numReductions = numReductions
        self.nModules = nModules
        self.nChannels = nChannels
        self.poolKernel = poolKernel
        self.poolStride = poolStride
        self.upSampleKernel = upSampleKernel
        """
        For the skip connection, a residual module (or sequence of residuaql modules)
        """

        _skip = []
        for _ in range(self.nModules):
            _skip.append(Residual(self.nChannels, self.nChannels))

        self.skip = nn.Sequential(*_skip)

        """
        First pooling to go to smaller dimension then pass input through
        Residual Module or sequence of Modules then  and subsequent cases:
            either pass through Hourglass of numReductions-1
            or pass through M.Residual Module or sequence of Modules
        """

        self.mp = nn.MaxPool2d(self.poolKernel, self.poolStride)

        _afterpool = []
        for _ in range(self.nModules):
            _afterpool.append(Residual(self.nChannels, self.nChannels))

        self.afterpool = nn.Sequential(*_afterpool)

        if (numReductions > 1):
            self.hg = Hourglass(self.nChannels, self.numReductions - 1, self.nModules, self.poolKernel, self.poolStride)
        else:
            _num1res = []
            for _ in range(self.nModules):
                _num1res.append(Residual(self.nChannels, self.nChannels))

            self.num1res = nn.Sequential(*_num1res)  # doesnt seem that important ?

        """
        Now another M.Residual Module or sequence of M.Residual Modules
        """

        _lowres = []
        for _ in range(self.nModules):
            _lowres.append(Residual(self.nChannels, self.nChannels))

        self.lowres = nn.Sequential(*_lowres)

        """
        Upsampling Layer (Can we change this??????)
        As per Newell's paper upsamping recommended
        """
        self.up = myUpsample()  # nn.Upsample(scale_factor = self.upSampleKernel)

    def forward(self, x):
        out1 = x
        out1 = self.skip(out1)
        out2 = x
        out2 = self.mp(out2)
        out2 = self.afterpool(out2)
        if self.numReductions > 1:
            out2 = self.hg(out2)
        else:
            out2 = self.num1res(out2)
        out2 = self.lowres(out2)
        out2 = self.up(out2)

        return out2 + out1


class StackedHourGlass(nn.Module):

    def __init__(self, nChannels, nStack, nModules, numReductions, nJoints):
        super(StackedHourGlass, self).__init__()
        self.nChannels = nChannels
        self.nStack = nStack
        self.nModules = nModules
        self.numReductions = numReductions
        self.nJoints = nJoints

        self.start = BnReluConv(3, 64, kernelSize=7, stride=2, padding=3)

        self.res1 = Residual(64, 128)
        self.mp = nn.MaxPool2d(2, 2)
        self.res2 = Residual(128, 128)
        self.res3 = Residual(128, self.nChannels)

        _hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan = [], [], [], [], [], []

        for _ in range(self.nStack):
            _hourglass.append(Hourglass(self.nChannels, self.numReductions, self.nModules))
            _ResidualModules = []
            for _ in range(self.nModules):
                _ResidualModules.append(Residual(self.nChannels, self.nChannels))
            _ResidualModules = nn.Sequential(*_ResidualModules)
            _Residual.append(_ResidualModules)
            _lin1.append(BnReluConv(self.nChannels, self.nChannels))
            _chantojoints.append(nn.Conv2d(self.nChannels, self.nJoints, 1))
            _lin2.append(nn.Conv2d(self.nChannels, self.nChannels, 1))
            _jointstochan.append(nn.Conv2d(self.nJoints, self.nChannels, 1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin1 = nn.ModuleList(_lin1)
        self.chantojoints = nn.ModuleList(_chantojoints)
        self.lin2 = nn.ModuleList(_lin2)
        self.jointstochan = nn.ModuleList(_jointstochan)

    def forward(self, x):
        x = self.start(x)
        x = self.res1(x)
        x = self.mp(x)
        x = self.res2(x)
        x = self.res3(x)
        out = []

        for i in range(self.nStack):
            x1 = self.hourglass[i](x)
            x1 = self.Residual[i](x1)
            x1 = self.lin1[i](x1)
            out.append(self.chantojoints[i](x1))
            x1 = self.lin2[i](x1)
            x = x + x1 + self.jointstochan[i](out[i])

        return (out)
