#!/usr/bin/env python

"""
caffe_cpm.py: Script for testing CPMs original implementation with
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
import sys


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
    im_nopad = cv.resize(im, (0, 0), fx=s, fy=s, interpolation=cv.INTER_CUBIC)

    # Padded to become multiple of 8 (downsize factor of the CPM)
    im, pad = util.padRightDownCorner(im)

    return im, im_nopad


def blend_map(im, heatmap):
    """
    Blends the detected heatmap with the original image
    :param im: np.array - original image
    :param heatmap: np.array - heatmap
    :return: original image and its heatmap blended
    """
    blend = np.uint8(np.clip(util.colorize(heatmap) * 0.5 + im * 0.5, 0, 255))

    return blend


def display_person_map(im, heatmap, coords):
    """
    Plots detected coordinates over the original image blended with
    its corresponding person heatmap that has been predicted.
    :param im: np.array - original image
    :param heatmap: np.array - person heatmap
    :param coords: detected coords
    """
    blend = blend_map(im, heatmap)

    plt.figure()

    plt.subplot(121)
    plt.title("Original image")
    plt.imshow(cv.cvtColor(blend, cv.COLOR_BGR2RGB))

    plt.subplot(122)
    plt.title("Person heatmap")
    plt.imshow(heatmap, cmap="gray")
    for x, y in coords:
        plt.plot(x, y, "x", 20, color="red")

    plt.show()


def display_pose_map(im, heatmaps):
    """
    Plots detected coordinates over the original image blended with
    its corresponding pose heatmap that has been predicted.
    :param im: np.array - original image
    :param heatmaps: np.array - pose heatmaps
    """
    plt.figure()

    for i, heatmap in enumerate(heatmaps):
        blend = blend_map(im, heatmap)
        plt.subplot(3, 5, i + 1)
        plt.imshow(cv.cvtColor(blend, cv.COLOR_BGR2RGB))

    plt.show()


def map_resize(new_shape, heatmap):
    """
    Resizes the heatmap detected.
    :param new_shape: tuple - target shape
    :param heatmap: np.array - heatmap
    :return: np.array - resized heatmap
    """
    # Resizes the output back to the size of the test image
    scale_y = new_shape[0] / float(heatmap.shape[0])
    scale_x = new_shape[1] / float(heatmap.shape[1])
    map_resized = cv.resize(heatmap, (0, 0), fx=scale_x, fy=scale_y,
                            interpolation=cv.INTER_CUBIC)

    return map_resized


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

def display_joints(im, joints):
    """
    Display predicted joints over the original image
    :param im: np.array - original image
    :param joints: list - joint coordinates
    """
    plt.figure()
    plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    for y, x in joints:
        plt.plot(x, y, "*", color="red")
    plt.show()

def display_result(im):
    """
    Display the resultant image
    :param im: np.array - resultant image
    """
    plt.figure()
    plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    plt.show()

def predict(deploy_models, im_original, viz):
    """
    Make a complete human pose estimation with Caffe CPM
    :param deploy_models: list - models & weights
    :param im_original: np.array - input image
    :param viz: bool - flag for visualizations
    :return: final image and joint coordinates
    """

    """
    Gets test image
    """
    boxsize = model['boxsize']  # image size used during training
    im, im_nopad = get_sample_ready(im_original, boxsize)

    """
    Person detection
    """
    person_detector = PersonDetector(deploy_models[0][0], deploy_models[0][1])

    start_time = time.time()
    person_map = person_detector.detect(im)
    print("\n\tPerson net took %.2f ms." % (1000 * (time.time() - start_time)))

    person_map_resized = map_resize(im.shape, person_map)
    person_coords = person_detector.peaks_coords(person_map_resized)
    print("\tPerson coordinates: ")
    for x, y in person_coords:
        print("\t\t(x,y)=" + str(x) + "," + str(y))

    if viz:
        display_person_map(im, person_map_resized, person_coords)

    """
    Pose estimation
    """
    pose_estimator = PoseEstimator(deploy_models[1][0], deploy_models[1][1])

    im_humans = pose_estimator.get_boxes(im, person_coords, boxsize)
    if viz:
        display_boxes(im_humans)

    pose_maps = []
    pose_coords = []
    for im_human, person_coord in zip(im_humans, person_coords):
        gauss_map = pose_estimator.gen_gaussmap(boxsize, model["sigma"])

        start_time = time.time()
        pose_map = pose_estimator.estimate(im_human, person_coords, gauss_map)
        print("\n\tPose net took %.2f ms." % (1000 * (time.time()
                                                        - start_time)))

        pose_map_resized = []
        joint_coords = []
        for joint_map in pose_map:
            # Resizes joint heatmaps
            joint_map_resized = map_resize(im_human.shape, joint_map)
            pose_map_resized.append(joint_map_resized)

            # Get resized coordinates of every joint
            joint_coord = pose_estimator.get_coords(joint_map_resized,
                                                    person_coord, boxsize)
            joint_coords.append(joint_coord)

        pose_coords.append(joint_coords)
        pose_maps.append(pose_map_resized)

        if viz:
            display_joints(im_original, joint_coords)

    # Draw limbs
    limbs = model["limbs"]
    im_final = pose_estimator.draw_limbs(limbs, im_original, pose_coords)

    if viz:
        display_result(im_final)

    return pose_coords, im_final

def load_model():
    """
    Get model and weights for pose and person detection
    :return: list - models & weights
    """
    param, model = config_reader()

    deploy_model_person = model['deployFile_person']
    weights_person = model['caffemodel_person']
    print("\nPerson detector:\n\tModel: " + deploy_model_person)
    print("\tWeights: " + weights_person)

    deploy_model = model['deployFile']
    weights = model['caffemodel']
    print("\nPose estimator:\n\tModel: " + deploy_model_person)
    print("\tWeights: " + weights_person)

    return [[deploy_model_person, weights_person], [deploy_model, weights]]


if __name__ == '__main__':
    caffe.set_mode_cpu()

    print("\n----Script for testing original CPMs repo (Caffe)----")

    im_path = sys.argv[1]
    im_original = cv.imread(im_path)

    viz = False
    models = load_model()
    pose_coords, im_predicted = predict(models, im_original, viz)

    plt.figure()
    plt.imshow(cv.cvtColor(im_predicted, cv.COLOR_BGR2RGB))
    plt.show()
