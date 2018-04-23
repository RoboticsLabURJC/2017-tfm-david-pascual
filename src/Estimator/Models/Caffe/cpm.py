#!/usr/bin/env python

"""
cpm.py: Script for testing CPMs original implementation with
Caffe. Based on @shihenw code:
https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/python/demo.ipynb
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/10/27"

import os
import sys
import time
import yaml
from matplotlib import pyplot as plt

# Avoids verbosity when loading Caffe model
os.environ["GLOG_minloglevel"] = "2"

import caffe
import cv2 as cv
import numpy as np

import util
from human_detector import HumanDetector
from pose_estimator import PoseEstimator


def set_dev(gpu):
    """
    Set Caffe to run on GPU or CPU.
    @param gpu: str - flag for GPU usage
    """
    if gpu:
        print("Using GPU...")
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        print("Using CPU...")
        caffe.set_mode_cpu()


def get_sample_ready(im, boxsize):
    """
    Processes the input image until it can be feed to the Caffe models
    @param im: np.array - input image
    @param boxsize: int - image size used during training
    @return: np.array - processed image
    """
    # The image is resized to conveniently crop a region that can
    # nicely fit a person
    s = boxsize / float(im.shape[0])
    im_nopad = cv.resize(im, None, fx=s, fy=s, interpolation=cv.INTER_CUBIC)

    # Padded to become multiple of 8 (downsize factor of the CPM)
    im_padded, pad = util.padRightDownCorner(im_nopad)

    return im_padded, im_nopad, s


def blend_map(im, heatmap):
    """
    Blends the detected heatmap with the original image
    @param im: np.array - original image
    @param heatmap: np.array - heatmap
    @return: original image and its heatmap blended
    """
    blend = np.uint8(np.clip(util.colorize(heatmap) * 0.5 + im * 0.5, 0, 255))

    return blend


def display_person_map(im, heatmap, coords):
    """
    Plots detected coordinates over the original image blended with
    its corresponding person heatmap that has been predicted.
    @param im: np.array - original image
    @param heatmap: np.array - person heatmap
    @param coords: detected coords
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
    @param im: np.array - original image
    @param heatmaps: np.array - pose heatmaps
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
    @param new_shape: tuple - target shape
    @param heatmap: np.array - heatmap
    @return: np.array - resized heatmap
    """
    # Resizes the output back to the size of the test image
    scale_y = new_shape[0] / float(heatmap.shape[0])
    scale_x = new_shape[1] / float(heatmap.shape[1])
    map_resized = cv.resize(heatmap, None, fx=scale_x, fy=scale_y,
                            interpolation=cv.INTER_CUBIC)

    return map_resized


def display_boxes(ims):
    """
    Displays the boxes that contain the detected humans.
    @param ims: np.array - images of each human
    """
    plt.figure()
    for i, im in enumerate(ims):
        plt.subplot(1, ims.shape[0], i + 1)
        plt.imshow(cv.cvtColor(np.uint8(im), cv.COLOR_BGR2RGB))
    plt.show()


def display_joints(im, joints):
    """
    Display predicted joints over the original image.
    @param im: np.array - original image
    @param joints: list - joint coordinates
    """
    plt.figure()
    plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    for y, x in joints:
        plt.plot(x, y, "*", color="red")
    plt.show()


def display_result(im):
    """
    Display the resultant image
    @param im: np.array - resultant image
    """
    plt.figure()
    plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    plt.show()


def load_model(models):
    """
    Get models and weights for pose and person detection.
    @param models: dict - Caffe models configuration
    @return: human detector & pose estimator
    """
    deploy_model_person = (models["deploy_human"], models["model_human"])
    print("\nPerson detector:\n\tEstimator: " + deploy_model_person[0])
    print("\tWeights: " + deploy_model_person[1])
    human_detector = HumanDetector(deploy_model_person[0],
                                   deploy_model_person[1])


    deploy_model_pose = (models["deploy_pose"], models["model_pose"])
    print("\nPose estimator:\n\tEstimator: " + deploy_model_pose[0])
    print("\tWeights: " + deploy_model_pose[1])
    pose_estimator = PoseEstimator(deploy_model_pose[0], deploy_model_pose[1])

    return human_detector, pose_estimator


def predict(im, data, models, boxsize=None, viz=True):
    """
    Make a complete human pose estimation with Caffe CPM.
    @param im: np.array - input image
    @param data: dict - models info
    @param models: list - human detector & pose estimator
    @param viz: bool - flag for visualizations
    @return: np.array, np.array, tuple - joint coordinates, result
    image & estimation times
    """
    if not boxsize:
        boxsize = data["boxsize"]

    human_detector, pose_estimator = models

    """
    Get test image
    """
    im_bsize, im_nopad, rate = get_sample_ready(im, boxsize)

    """
    Person detection
    """
    start_t = time.time()
    person_map = human_detector.detect(im_bsize)
    human_t = int(1000 * (time.time() - start_t))
    print("\nPerson net took %.2f ms." % human_t)

    person_map_resized = map_resize(im_bsize.shape, person_map)
    person_coords = human_detector.peaks_coords(person_map_resized)
    print("Person coordinates: ")
    for x, y in person_coords:
        print("\t\t(x,y)=" + str(x) + "," + str(y))

    if viz:
        display_person_map(im_bsize, person_map_resized, person_coords)

    """
    Pose estimation
    """
    im_humans = pose_estimator.get_boxes(im_bsize, person_coords, boxsize)
    if viz:
        display_boxes(im_humans)

    pose_maps = []
    pose_coords = []
    if len(im_humans):
        pose_t = 0
    else:
        pose_t = ""
    for im_human, person_coord in zip(im_humans, person_coords):
        gauss_map = pose_estimator.gen_gaussmap(boxsize, data["sigma"])

        start_t = time.time()
        pose_map = pose_estimator.estimate(im_human, gauss_map)
        pose_t += int(1000 * (time.time() - start_t))

        print("\nPose net took %.2f ms." % int(1000 * (time.time() - start_t)))

        pose_map_resized = []
        joint_coords = []
        for joint_map in pose_map:
            # Resizes joint heatmaps
            joint_map_resized = map_resize(im_human.shape, joint_map)
            pose_map_resized.append(joint_map_resized)

            # Get resized coordinates of every joint
            joint_coord = pose_estimator.get_coords(joint_map_resized,
                                                    person_coord, rate,
                                                    boxsize)
            joint_coords.append(joint_coord)

        pose_coords.append(joint_coords)
        pose_maps.append(pose_map_resized)

        if viz:
            display_joints(im, joint_coords)

    # Draw limbs
    limbs = data["limbs"]
    colors = data["colors"]
    im_final = pose_estimator.draw_limbs(im, pose_coords, limbs, colors)

    if viz:
        display_result(im_final)

    return im_final, pose_coords, (human_t, pose_t)


if __name__ == "__main__":
    print("\n----Script for testing original CPMs repo (Caffe)----")

    im_path = sys.argv[1]
    im_original = cv.imread(im_path)

    data = None
    with open("cpm.yml", "r") as stream:
        try:
            data = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    set_dev(data)
    deploy_models = load_model(data["caffe_models"])

    start_time = time.time()
    im_predicted, _, _ = predict(im_original, data, deploy_models)
    print("\nTotal time (w/out loading) %.2f ms." % (1000 * (time.time()
                                                             - start_time)))
    plt.figure()
    plt.imshow(cv.cvtColor(im_predicted, cv.COLOR_BGR2RGB))
    plt.show()
