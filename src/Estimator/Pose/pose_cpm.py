#!/usr/bin/env python

"""
pose_cpm.py: Convolutional Pose Machines.
Based on @shihenw code:
https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/python/demo.ipynb
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/05/22"

import math
import os

# Avoids verbosity when loading Caffe model
os.environ["GLOG_minloglevel"] = "2"

import caffe
import cv2
import numpy as np

from pose import PoseEstimator


def crop_human(sample, c, s, bsize):
    """
    Crop human in the image depending on subject center and scale.
    @param sample: np.array - input image
    @param c: list - approx. human center
    @param s: float - approx. human scale wrt 200px
    @param bsize: int - boxsize
    @return: np.array - cropped human
    """
    cx, cy = c

    # Resize image and center according to given scale
    im_resized = cv2.resize(sample, None, fx=s, fy=s)

    h, w, d = im_resized.shape

    pad_up = int(bsize / 2 - cy)
    pad_down = int(bsize / 2 - (h - cy))
    pad_left = int(bsize / 2 - cx)
    pad_right = int(bsize / 2 - (w - cx))

    # Apply padding or crop image as needed
    if pad_up > 0:
        pad = np.ones((pad_up, w, d), np.uint8) * 128
        im_resized = np.vstack((pad, im_resized))
    else:
        im_resized = im_resized[-pad_up:, :, :]
    h, w, d = im_resized.shape

    if pad_down > 0:
        pad = np.ones((pad_down, w, d), np.uint8) * 128
        im_resized = np.vstack((im_resized, pad))
    else:
        im_resized = im_resized[:h + pad_down, :, :]
    h, w, d = im_resized.shape

    if pad_left > 0:
        pad = np.ones((h, pad_left, d), np.uint8) * 128
        im_resized = np.hstack((pad, im_resized))
    else:
        im_resized = im_resized[:, -pad_left:, :]
    h, w, d = im_resized.shape

    if pad_right > 0:
        pad = np.ones((h, pad_right, d), np.uint8) * 128
        im_resized = np.hstack((im_resized, pad))
    else:
        im_resized = im_resized[:, :w + pad_right, :]

    return im_resized


def map_resize(new_shape, heatmap):
    # Resizes the output back to the size of the test image
    scale_y = new_shape[0] / float(heatmap.shape[0])
    scale_x = new_shape[1] / float(heatmap.shape[1])
    map_resized = cv2.resize(heatmap, None, fx=scale_x, fy=scale_y,
                             interpolation=cv2.INTER_CUBIC)

    return map_resized


class PoseCPM(PoseEstimator):
    def __init__(self, model_fname, boxsize, sigma, confidence_th=0.3):
        """
        Constructs Estimator class.
        @param model_fname: Caffe models
        @param weights: Caffe models weights
        """
        PoseEstimator.__init__(self, model_fname, boxsize, confidence_th)
        self.model, self.weights = self.model_fname
        self.sigma = sigma
        self.gauss_map = self.gen_gaussmap()

    def init_net(self):
        caffe.set_mode_gpu()
        self.net = caffe.Net(self.model, self.weights, caffe.TEST)

    def estimate(self):
        """
        Estimates human pose.
        @param im: np.array - input image
        @param gaussmap: np.array - Gaussian map
        @return: np.array: articulations coordinates
        """
        if not self.net:
            self.init_net()

        # Adds gaussian map channel to the input
        input_4ch = np.ones((self.im.shape[0], self.im.shape[1], 4))
        input_4ch[:, :, 0:3] = self.im / 256.0 - 0.5  # normalize to [-0.5, 0.5]
        input_4ch[:, :, 3] = self.gauss_map

        # Adapts input to the net
        input_adapted = np.transpose(np.float32(input_4ch[:, :, :, np.newaxis]),
                                     (3, 2, 0, 1))
        self.net.blobs['data'].reshape(*input_adapted.shape)
        self.net.blobs['data'].data[...] = input_adapted

        # Estimates the pose
        output_blobs = self.net.forward()
        pose_map = np.squeeze(self.net.blobs[output_blobs.keys()[0]].data)

        return pose_map

    def gen_gaussmap(self):
        """
        Generates a grayscale image with a centered Gaussian
        @param sigma: float - Gaussian sigma
        @return: np.array - Gaussian map
        """
        gaussmap = np.zeros((self.boxsize, self.boxsize, 1))
        for x in range(self.boxsize):
            for y in range(self.boxsize):
                dist_sq = (x - self.boxsize / 2) * (x - self.boxsize / 2) \
                          + (y - self.boxsize / 2) * (y - self.boxsize / 2)
                exponent = dist_sq / 2.0 / self.sigma / self.sigma
                gaussmap[y, x, :] = math.exp(-exponent)

        return np.squeeze(gaussmap)

    def get_coords(self, sample, human_bbox):
        """
        Estimate human pose given an input image.
        @param sample: np.array - original input image
        @param human: np.array - cropped human image
        @param config: dict - CPM settings
        @param model: pose estimator object
        @param c: np.array - human center
        @param s: int - human scale
        @param viz: bool - flag for joint visualization
        @return: np.array - joint coords
        """
        caffe.set_mode_gpu()

        (ux, uy), (lx, ly) = human_bbox

        # Get scale
        scale = float(self.boxsize) / sample.shape[0]

        # Get center
        cx, cy = (int((ux + lx) * scale / 2), int((uy + ly) * scale / 2))

        im_human = crop_human(sample, (cx, cy), scale, self.boxsize)

        self.im = im_human.copy()

        pose_map = self.estimate()

        joint_coords = []
        for joint_map in pose_map:
            joint_map_resized = map_resize(self.im.shape, joint_map)

            # Find joint heatmap maxima
            joint = [-1, -1]
            if joint_map_resized.max() >= self.confidence_th:
                joint = list(np.unravel_index(joint_map_resized.argmax(),
                                              joint_map_resized.shape))

                # Back to full coordinates
                joint[0] = (joint[0] - (self.boxsize / 2) + cy) / scale
                joint[1] = (joint[1] - (self.boxsize / 2) + cx) / scale

            joint_coords.append(joint)

        joint_coords = np.array([[int(x), int(y)] for y, x in joint_coords])

        return joint_coords
