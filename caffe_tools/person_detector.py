#!/usr/bin/env python

"""
person_detector.py: Module for detecting persons with Caffe.
Based on @shihenw code:
https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/python/demo.ipynb
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/01"

# Inserts path to pycaffe
import sys

sys.path.insert(0, "/home/dpascualhe/caffe/python")

import caffe
import time
import numpy as np
import scipy

class PersonDetector():
    """
    Class for person detection.
    """

    def __init__(self, model, weights):
        """
        Class constructor.
        :param model: caffe model
        :param weights: caffe model weights
        """

        # Reshapes the model input accordingly
        self.net = caffe.Net(model, weights, caffe.TEST)

    def detect(self, im):
        """
        Detects people in the image.
        :param im: np.array - input image
        :return: np.array - heatmap
        """
        # Reshapes and normalizes the input image
        im = np.float32(im[:, :, :, np.newaxis])
        im = np.transpose(im, (3, 2, 0, 1)) / 256 - 0.5
        self.net.blobs['image'].reshape(*im.shape)

        # Feeds the net
        self.net.blobs['image'].data[...] = im

        # Person detection
        start_time = time.time()
        output_blobs = self.net.forward()

        map = np.squeeze(self.net.blobs[output_blobs.keys()[0]].data)

        return map

    def peaks_coords(self, heatmap):
        """
        Gets the exact coordinates of each person in the heatmap.
        :param heatmap: np.array - heatmap
        :return: np.array - people coordinates
        """
        # Founds the peaks in the output
        data_max = scipy.ndimage.filters.maximum_filter(heatmap, 3)
        max = (heatmap == data_max)
        thresh = (data_max > 0.5)
        max[thresh == 0] = 0

        # Peaks coordinates
        x = np.nonzero(max)[1]
        y = np.nonzero(max)[0]
        peaks_coords = []
        for x_coord, y_coord in zip(x, y):
            peaks_coords.append([x_coord, y_coord])

        return np.array(peaks_coords)
