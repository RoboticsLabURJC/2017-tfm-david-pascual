#!/usr/bin/env python

"""
cpm.py: Script for testing CPMs implementation in TensorFlow.
Based on @psycharo code:
https://github.com/psycharo/cpm/blob/master/example.ipynb
"""

__author__ = "David Pascual Hernandez"
__date__ = "2017/12/26"

import sys
import time
import yaml
from matplotlib import pyplot as plt

import cv2
import numpy as np
import tensorflow as tf

from human_detector import HumanDetector
from pose_estimator import PoseEstimator

def load_model(data, human_shape, config):
    """
    Get human & pose models.
    @param data: models paths
    @return: human & pose models
    """
    boxsize = data["boxsize"]
    pose_shape = (boxsize, boxsize, 15)

    human_path = data["tf_models"]["model_human"]
    print(human_path)
    human_detector = HumanDetector(human_shape, config, human_path, boxsize)

    pose_path = data["tf_models"]["model_pose"]
    pose_estimator = PoseEstimator(pose_shape, config, pose_path, boxsize)

    return human_detector, pose_estimator


def set_dev(gpu):
    """
    GPU settings.
    @param gpu: flag for GPU usage
    @return: TensorFlow configuration
    """
    if gpu:
        dev = {"GPU": 1}
    else:
        dev = {"CPU": 1, "GPU": 0}

    config = tf.ConfigProto(device_count=dev, allow_soft_placement=True,
                            log_device_placement=False)

    return config


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
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

    # Padded to become multiple of 8 (downsize factor of the CPM)
    # im_padded, pad = util.padRightDownCorner(im_nopad)

    return im, s


def predict(im, models, data, viz=True):
    """
    Estimates human pose given an image.
    @param im: np.array - image
    @param models: TensorFlow human & pose models
    @param data: parsed YAML config. file
    @param viz: bool - Flag for visualization
    @return: np.array, np.array, tuple - joint coordinates, result
    image & estimation times
    """
    tf.reset_default_graph()

    human_detector, pose_estimator = models

    full_coords_joints = []

    """
    Human detection
    """
    human_detector.im = im

    start_t = time.time()
    full_map_human = human_detector.get_map()
    human_t = int(1000 * (time.time() - start_t))

    if viz:
        plt.figure(), plt.imshow(np.squeeze(full_map_human)), plt.show()

    humans_coords = human_detector.get_humans_coords(full_map_human)
    for x, y in humans_coords:
        print("\t\t(x,y)=" + str(x) + "," + str(y))

    pose_t = ""
    if len(humans_coords):
        im_humans = human_detector.crop_humans(humans_coords)
        map_gaussian = human_detector.gen_gaussmap(data["sigma"])

        pose_input = []
        for im_human in im_humans:
            if viz:
                im_viz = np.uint8((im_human + 0.5) * 255)
                plt.figure(), plt.imshow(im_viz), plt.show()
            pose_input.append(np.dstack((np.squeeze(im_human), map_gaussian)))

        pose_input = np.array(pose_input)

        """
        Pose estimation
        """
        pose_estimator.humans = pose_input

        start_t = time.time()
        full_maps_joints = pose_estimator.get_map()
        pose_t = int(1000 * (time.time() - start_t))

        if len(humans_coords) > 1:
            for i, map_joints in enumerate(full_maps_joints):
                if viz:
                    plt.figure()
                    for j, map_joint in enumerate(map_joints):
                        plt.subplot(3, 5, 1 + j), plt.imshow(map_joint)
                    plt.show()

                coords_joints = pose_estimator.get_joints(map_joints,
                                                          humans_coords[i],
                                                          human_detector.factor)

                if viz:
                    for x, y in coords_joints:
                        cv2.circle(im, (int(x), int(y)), 2, (0, 0, 255), 2)

                im = pose_estimator.draw_limbs(im, coords_joints, data)
                full_coords_joints.append(coords_joints)
        else:
            if viz:
                plt.figure()
                for j, map_joint in enumerate(full_maps_joints):
                    plt.subplot(3, 5, 1 + j), plt.imshow(map_joint)
                plt.show()

            coords_joints = pose_estimator.get_joints(full_maps_joints,
                                                      humans_coords[0],
                                                      human_detector.factor)

            if viz:
                for x, y in coords_joints:
                    cv2.circle(im, (int(x), int(y)), 2, (0, 0, 255), 2)

            im = pose_estimator.draw_limbs(im, coords_joints, data)
            full_coords_joints.append(coords_joints)

    return im, np.array(full_coords_joints), (human_t, pose_t)


if __name__ == "__main__":
    sample_path = sys.argv[1]
    sample = cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB)

    data = None
    with open("cpm.yml", "r") as stream:
        try:
            data = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    tf_config = set_dev(data)
    models = load_model(data, sample.shape, tf_config)

    im_estimated, _, _ = predict(sample, models, data)

    plt.figure()
    plt.imshow(im_estimated)
    plt.show()
