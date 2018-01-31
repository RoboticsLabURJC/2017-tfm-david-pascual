#!/usr/bin/env python

"""
cpm.py: Script for testing CPMs implementation in TensorFlow.
Based on @psycharo code:
https://github.com/psycharo/cpm/blob/master/example.ipynb
"""

__author__ = "David Pascual Hernandez"
__date__ = "2017/12/26"

import sys
import yaml
from matplotlib import pyplot as plt

import cv2
import numpy as np
import tensorflow as tf

from tools.human_detector import HumanDetector
from tools.pose_estimator import PoseEstimator


def load_model(data):
    """
    Get person & pose models paths.
    @param data: models paths
    @return: models paths
    """
    human_path = data["model_human"]
    pose_path = data["model_pose"]

    return human_path, pose_path


def set_dev(data):
    """
    GPU settings.
    @param data: parsed YAML config. file
    @return: TensorFlow configuration
    """
    if data["GPU"]:
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


def predict(im, config, models, data, viz=True):
    """
    Estimates human pose given an image.
    @param im: np.array - image
    @param config: TensorFlow config
    @param models: TensorFlow human & pose models
    @param data: parsed YAML config. file
    @param viz: bool - Flag for visualization
    @return: np.array, np.array - joint coordinates & final image
    """
    tf.reset_default_graph()

    human_model, pose_model = models

    full_coords_joints = []

    bsize = data["boxsize"]

    """
    Human detection
    """
    human_detector = HumanDetector(im, config, human_model, bsize)

    full_map_human = human_detector.get_map()
    if viz:
        plt.figure(), plt.imshow(np.squeeze(full_map_human)), plt.show()

    humans_coords = human_detector.get_humans_coords(full_map_human)
    for x, y in humans_coords:
        print("\t\t(x,y)=" + str(x) + "," + str(y))

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
        pose_estimator = PoseEstimator(pose_input, config, pose_model, bsize)

        full_maps_joints = pose_estimator.get_map()

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

                im = pose_estimator.draw_limbs(im, coords_joints)
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

    return im, np.array(full_coords_joints)


if __name__ == "__main__":
    sample_path = sys.argv[1]
    sample = cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB)

    data = None
    with open("cpm.yml", "r") as stream:
        try:
            data = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    model_paths = load_model(data["tf_models"])

    tf_config = set_dev(data)

    im_estimated, _ = predict(sample, tf_config, model_paths, data)

    plt.figure()
    plt.imshow(im_estimated)
    plt.show()
