#!/usr/bin/env python

"""
human_detector.py: Module for detecting persons with Caffe.
Based on @shihenw code:
https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/python/demo.ipynb
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/01"

import caffe
import numpy as np
import scipy


class HumanDetector:
    """
    Class for person detection.
    """

    def __init__(self, model, weights):
        """
        Class constructor.
        @param model: caffe model
        @param weights: caffe model weights
        """

        # Reshapes the model input accordingly
        self.net = caffe.Net(model, weights, caffe.TEST)

    def detect(self, im):
        """
        Detects people in the image.
        @param im: np.array - input image
        @return: np.array - heatmap
        """
        # Reshapes and normalizes the input image
        im = np.float32(im[:, :, :, np.newaxis])
        im = np.transpose(im, (3, 2, 0, 1)) / 256 - 0.5
        self.net.blobs['image'].reshape(*im.shape)

        # Feeds the net
        self.net.blobs['image'].data[...] = im

        # Person detection
        output_blobs = self.net.forward()

        humans_map = np.squeeze(self.net.blobs[output_blobs.keys()[0]].data)

        return humans_map

    @staticmethod
    def peaks_coords(heatmap):
        """
        Gets the exact coordinates of each person in the heatmap.
        @param heatmap: np.array - heatmap
        @return: np.array - people coordinates
        """
        # Founds the peaks in the output
        # noinspection PyUnresolvedReferences
        data_max = scipy.ndimage.filters.maximum_filter(heatmap, 3)
        peaks = (heatmap == data_max)
        thresh = (data_max > 0.5)
        peaks[thresh == 0] = 0

        # Peaks coordinates
        x = np.nonzero(peaks)[1]
        y = np.nonzero(peaks)[0]
        peaks_coords = []
        for x_coord, y_coord in zip(x, y):
            peaks_coords.append([x_coord, y_coord])

        return np.array(peaks_coords)
