#!/usr/bin/env python

"""
human_caffe.py: Module for detecting persons with Caffe.
Based on @shihenw code:
https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/python/demo.ipynb
"""

import os

# Avoids verbosity when loading Caffe model
os.environ["GLOG_minloglevel"] = "2"

import caffe
import cv2
import numpy as np

from human import Human

__author__ = "David Pascual Hernandez"
__date__ = "2018/05/22"


def map_resize(new_shape, heatmap):
    # Resizes the output back to the size of the test image
    scale_y = new_shape[0] / float(heatmap.shape[0])
    scale_x = new_shape[1] / float(heatmap.shape[1])
    map_resized = cv2.resize(heatmap, None, fx=scale_x, fy=scale_y,
                             interpolation=cv2.INTER_CUBIC)

    return map_resized


class HumanDetector(Human):
    """
    Class for person detection.
    """

    def __init__(self, model, boxsize):
        """
        Class constructor.
        @param model: caffe models
        @param weights: caffe models weights
        """
        Human.__init__(self, boxsize)

        # Reshapes the models input accordingly
        self.model, self.weights = model
        self.net = None

    def init_net(self):
        caffe.set_mode_gpu()
        self.net = caffe.Net(self.model, self.weights, caffe.TEST)

    def detect(self):
        """
        Detects people in the image.
        @param im: np.array - input image
        @return: np.array - heatmap
        """
        if not self.net:
            self.init_net()

        # Reshapes and normalizes the input image
        im = np.float32(self.im[:, :, :, np.newaxis])
        im = np.transpose(im, (3, 2, 0, 1)) / 256 - 0.5
        self.net.blobs['image'].reshape(*im.shape)

        # Feeds the net
        self.net.blobs['image'].data[...] = im

        # Person detection
        output_blobs = self.net.forward()

        self.heatmap = np.squeeze(self.net.blobs[output_blobs.keys()[0]].data)
        self.heatmap = map_resize(self.im.shape, self.heatmap)
