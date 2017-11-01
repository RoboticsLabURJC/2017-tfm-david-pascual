#!/usr/bin/env python

"""
caffe_test.py: Script for testing CPMs original implementation with
Caffe. Based on @shihenw code:
https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/python/demo.ipynb
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/10/2017"

# Inserts path to pycaffe
import sys

sys.path.insert(0, "/home/dpascualhe/caffe/python")

# Avoids verbosity when loading Caffe model
import os

os.environ['GLOG_minloglevel'] = '2'

import caffe
from caffe_tools import util
from caffe_tools.config_reader import config_reader
from caffe_tools.person_detector import PersonDetector
from caffe_tools.pose_estimator import PoseEstimator
import cv2 as cv
from matplotlib import pyplot as plt
import time
import numpy as np


def get_sample_ready(im, boxsize):
    """
    Processes the input image until it can be feed to the Caffe model
    :param im: np.array - input image
    :param boxsize: int - image size used during training
    :return: np.array - processed image
    """
    # The image is resized to conveniently crop a region that can
    # nicely fit a person
    s = boxsize / float(im.shape[0])
    im = cv.resize(im, (0, 0), fx=s, fy=s, interpolation=cv.INTER_CUBIC)

    # Padded to become multiple of 8 (downsize factor of the CPM)
    im, pad = util.padRightDownCorner(im)

    return im


def display_map(im, map, coords):
    """
    Plots detected coordinates over the original image blended with
    its corresponding heatmap that has been predicted
    :param im: np.array - original image
    :param map: np.array - detected heatmap
    :param coords: detected coords
    """
    blend = np.uint8(np.clip(util.colorize(map) * 0.5 + im * 0.5, 0, 255))

    plt.figure()

    plt.subplot(121)
    plt.title("Original image")
    plt.imshow(cv.cvtColor(blend, cv.COLOR_BGR2RGB))

    plt.subplot(122)
    plt.title("Person heatmap")
    plt.imshow(person_map_resized, cmap="gray")
    for x, y in coords:
        plt.plot(x, y, "x", 20, color="red")

    plt.show()


def display_boxes(ims):
    """
    Displays the boxes that contain the detected humans.
    :param ims: np.array - images of each human
    """
    plt.figure()
    for i, im in enumerate(ims):
        plt.subplot(1, ims.shape[0], i + 1)
        plt.imshow(cv.cvtColor(np.uint8(im), cv.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
    caffe.set_mode_cpu()

    print("\n----Script for testing original CPMs repo (Caffe)----")

    """
    Gets test image
    """
    im_path = "samples/nadal.png"
    im_original = cv.imread(im_path)
    print("Loaded: " + im_path)

    param, model = config_reader()
    boxsize = model['boxsize']  # image size used during training
    im = get_sample_ready(im_original, boxsize)

    """
    Person detection
    """
    deploy_model_person = model['deployFile_person']
    weights_person = model['caffemodel_person']
    print("\nPerson detector:\n\tModel: " + deploy_model_person)
    print("\tWeights: " + weights_person)

    person_detector = PersonDetector(deploy_model_person, weights_person)

    start_time = time.time()
    person_map = person_detector.detect(im)
    print("\n\tPerson net took %.2f ms." % (1000 * (time.time() - start_time)))

    person_map_resized = person_detector.map_resize(im.shape, person_map)
    person_coords = person_detector.peaks_coords(person_map_resized)
    print("\tPerson coordinates: ")
    for x, y in person_coords:
        print("\t\t(x,y)=" + str(x) + "," + str(y))

    # display_map(im, person_map_resized, person_coords)

    """
    Pose estimation
    """
    deploy_model = model['deployFile']
    weights = model['caffemodel']
    print("\nPose estimator:\n\tModel: " + deploy_model_person)
    print("\tWeights: " + weights_person)

    pose_estimator = PoseEstimator(deploy_model, weights)

    humans = pose_estimator.get_boxes(im, person_coords, boxsize)
    # display_boxes(humans)

    for human in humans:
        start_time = time.time()
        gauss_map = pose_estimator.gen_gaussmap(boxsize, model["sigma"])
        print("\n\tPerson net took %.2f ms." % (1000 * (time.time() - start_time)))

        plt.figure()
        plt.imshow(np.uint8(np.clip(gauss_map, 0, 255)), cmap="gray")
        plt.show()

        # TODO fix pose estimator
