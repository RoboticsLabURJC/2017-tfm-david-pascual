#!/usr/bin/env python

"""
caffe_test.py: Script for testing CPMs original implementation with
Caffe. Based on @shihenw code:
https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/python/demo.ipynb
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/10/2017"

# Inserts paths to pycaffe and original CPMs repo modules
import sys

sys.path.insert(0, "/home/dpascualhe/caffe/python")
sys.path.insert(0, "convolutional-pose-machines-release/testing/python/")

import cv2 as cv
import util
from config_reader import config_reader

if __name__ == '__main__':
    """
    Getting the sample ready
    """
    sample = 'samples/nadal.png'
    im = cv.imread(sample)
    print("Loaded: " + sample)

    # The image is resized to fit the input
    param, model = config_reader()
    boxsize = model['boxsize']  # image size used during training
    scale = boxsize / float(im.shape[0])
    test_im = cv.resize(im, (0, 0), fx=scale, fy=scale,
                        interpolation=cv.INTER_CUBIC)

    # Padded to become multiple of 8 (downsize factor of the CPM)
    test_im, pad = util.padRightDownCorner(test_im)
    print("Image resized: " + str(im.shape) + " -> " + str(test_im.shape))

    # TODO load Caffe model
    # TODO estimate pose
