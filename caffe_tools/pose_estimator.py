#!/usr/bin/env python

"""
pose_estimator.py: Module for estimating human pose with Caffe.
Based on @shihenw code:
https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/python/demo.ipynb
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/01"

# Inserts path to pycaffe
import sys

sys.path.insert(0, "/home/dpascualhe/caffe/python")

import caffe
import numpy as np
import math


class PoseEstimator():
    def __init__(self, model, weights):
        """
        Constructs PoseEstimator class.
        :param model: Caffe model
        :param weights: Caffe model weights
        """
        self.net = caffe.Net(model, weights, caffe.TEST)

    def get_boxes(self, im, coords, size):
        """
        Crops the original image once for each person.
        :param im: np.array - input image
        :param coords: np.array - human coordinates
        :param size: int - size of the squared box
        :return: np.array - cropped images around each person
        """
        num_people = coords.shape[0]
        boxes = np.ones((num_people, size, size, 3)) * 128

        for i in range(num_people):
            for x_box in range(size):
                for y_box in range(size):
                    x_i = x_box - size / 2 + coords[i][0]
                    y_i = y_box - size / 2 + coords[i][1]
                    if (x_i >= 0 and x_i < im.shape[1]) \
                            and (y_i >= 0 and y_i < im.shape[0]):
                        boxes[i, y_box, x_box, :] = im[y_i, x_i, :]

        return boxes

    def gen_gaussmap(self, size, sigma):
        """
        Generates a grayscale image with a centered Gaussian
        :param size: int - map size
        :param sigma: float - Gaussian sigma
        :return: np.array - Gaussian map
        """
        gaussmap = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                dist_sq = (x - size / 2) * (x - size / 2) \
                          + (y - size / 2) * (y - size / 2)
                exponent = dist_sq / 2.0 / sigma / sigma
                gaussmap[y, x] = math.exp(-exponent)

        return gaussmap

    def estimate(self, im, coords, gaussmap):
        """
        Estimates human pose.
        :param im: np.array - input image
        :param coords: np.array - human coordinates
        :param gaussmap: np.array - Gaussian map
        :return: np.array: articulations coordinates
        """
        # Adds gaussian map channel to the input
        input_4ch = np.ones((im.shape[0], im.shape[1], 4))
        input_4ch[:, :, 0:3] = im / 256.0 - 0.5  # normalize to [-0.5, 0.5]
        input_4ch[:, :, 3] = gaussmap

        # Adapts input to the net
        input_adapted = np.transpose(np.float32(input_4ch[:, :, :, np.newaxis]),
                                     (3, 2, 0, 1))
        self.net.blobs['data'].data[...] = input_adapted

        # Estimates the pose
        output_blob = self.net.forward()

        return output_blob
