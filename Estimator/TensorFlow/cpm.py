#!/usr/bin/env python

"""
cpm.py: Script for testing CPMs implementation in TensorFlow.
Based on @psycharo code:
https://github.com/psycharo/cpm/blob/master/example.ipynb
"""

__author__ = "David Pascual Hernandez"
__date__ = "2017/12/26"

import os
import sys
from matplotlib import pyplot as plt

import cv2
import numpy as np
import tensorflow as tf

from tf_tools.human_detector import HumanDetector


def load_model(path):
    """
    Get person & pose model paths.
    @param path: folder path
    @return: model paths
    """
    human_path = os.path.join(path, "person_net.ckpt")
    pose_path = os.path.join(path, "pose_net.ckpt")

    return human_path, pose_path


def set_dev():
    """
    GPU settings.
    @return: TensorFlow configuration
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    return config


def predict(im, config, models, bsize, viz=True):
    """
    Estimates human pose given an image.
    @param im: np.array - image
    @param config: TensorFlow config
    @param models: TensorFlow human & pose models
    @param bsize: int - size of the human cropped region
    @param viz: bool - Flag for visualization
    @return: np.array, np.array - joint coordinates & final image
    """
    tf.reset_default_graph()

    """
    Human detection
    """
    human_model, pose_model = models

    human_detector = HumanDetector(im, config, human_model, bsize)

    full_map_human = human_detector.get_map()
    if viz:
        plt.figure(), plt.imshow(np.squeeze(full_map_human)), plt.show()

    humans_coords = human_detector.get_humans_coords(full_map_human)
    im_humans = human_detector.crop_humans(humans_coords)
    map_gaussian = human_detector.gen_gaussmap(21)

    pose_input = []
    for im_human in im_humans:
        if viz:
            plt.figure(), plt.imshow(np.squeeze(im_human)), plt.show()
        pose_input.append(np.dstack((np.squeeze(im_human), map_gaussian)))


if __name__ == '__main__':
    sample_path = sys.argv[1]
    sample = cv2.imread(sample_path)

    model_paths = load_model("model/")

    boxsize = 128
    tf_config = set_dev()

    predict(sample, tf_config, model_paths, boxsize)
