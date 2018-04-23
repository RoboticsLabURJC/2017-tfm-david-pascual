#!/usr/bin/env python

"""
gen_preds.py: Perform a quantitative analysis of CPMs.
"""

__author__ = "David Pascual Hernandez"
__date__ = "2017/02/24"

import argparse
import math
import os
import sys
import time
import yaml
from matplotlib import pyplot as plt

import cv2
import h5py
import numpy as np

sys.path.append("..")


def get_args():
    """
    Get program arguments and parse them.
    @return: dict - arguments
    """
    available_fw = ["caffe", "tensorflow"]

    ap = argparse.ArgumentParser(
        description="Store predictions for quantitative analysis")
    ap.add_argument("-f", "--framework", type=str, required=False,
                    default="caffe", choices=available_fw,
                    help="Framework to test")
    ap.add_argument("-s", "--subset", type=str, required=False,
                    default="../Datasets/v0/mpii_lsp_train.h5")
    ap.add_argument("-v", "--viz", type=bool, required=False,
                    default=False)

    return vars(ap.parse_args())


def crop_human(sample, c, s, bsize):
    """
    Crop human in the image depending on subject center and scale.
    @param sample: np.array - input image
    @param c: list - approx. human center
    @param s: float - approx. human scale wrt 200px
    @param bsize: int - boxsize
    @return: np.array - cropped human
    """
    cx, cy = c

    # Resize image and center according to given scale
    im_resized = cv2.resize(sample, None, fx=s, fy=s)

    h, w, d = im_resized.shape

    pad_up = int(bsize / 2 - cy)
    pad_down = int(bsize / 2 - (h - cy))
    pad_left = int(bsize / 2 - cx)
    pad_right = int(bsize / 2 - (w - cx))

    # Apply padding or crop image as needed
    if pad_up > 0:
        pad = np.ones((pad_up, w, d)) * 128
        im_resized = np.vstack((pad, im_resized))
    else:
        im_resized = im_resized[-pad_up:, :, :]
    h, w, d = im_resized.shape

    if pad_down > 0:
        pad = np.ones((pad_down, w, d)) * 128
        im_resized = np.vstack((im_resized, pad))
    else:
        im_resized = im_resized[:h + pad_down, :, :]
    h, w, d = im_resized.shape

    if pad_left > 0:
        pad = np.ones((h, pad_left, d)) * 128
        im_resized = np.hstack((pad, im_resized))
    else:
        im_resized = im_resized[:, -pad_left:, :]
    h, w, d = im_resized.shape

    if pad_right > 0:
        pad = np.ones((h, pad_right, d)) * 128
        im_resized = np.hstack((im_resized, pad))
    else:
        im_resized = im_resized[:, :w + pad_right, :]

    return im_resized


def load_caffe_model(models):
    """
    Load Caffe pose estimation model.
    @param models: dict - Caffe model & weights
    @return: pose estimator object
    """
    return PoseEstimator(models["deploy_pose"], models["model_pose"])


def load_tf_model(config, dev_config):
    """
    Load TensorFlow pose estimation model.
    @param models: dict - TensorFlow model & weights
    @param dev_config: TensorFlow device configuration object
    @return: pose estimator object
    """
    boxsize = config["boxsize"]
    models = config["tf_models"]
    pose_shape = (boxsize, boxsize, 15)

    return PoseEstimator(pose_shape, dev_config, models["model_pose"], boxsize)


def gen_gaussmap(boxsize, sigma):
    """
    Generates a grayscale image with a centered Gaussian
    @param sigma: float - Gaussian sigma
    @return: np.array - Gaussian map
    """
    gaussmap = np.zeros((boxsize, boxsize, 1))
    for x in range(boxsize):
        for y in range(boxsize):
            dist_sq = (x - boxsize / 2) * (x - boxsize / 2) \
                      + (y - boxsize / 2) * (y - boxsize / 2)
            exponent = dist_sq / 2.0 / sigma / sigma
            gaussmap[y, x, :] = math.exp(-exponent)

    return gaussmap


def estimate_caffe_pose(sample, human, config, model, c, s, viz):
    """
    Estimate human pose given an input image.
    @param sample: np.array - original input image
    @param human: np.array - cropped human image
    @param config: dict - CPM settings
    @param model: pose estimator object
    @param c: np.array - human center
    @param s: int - human scale
    @param viz: bool - flag for joint visualization
    @return: np.array - joint coords
    """
    gauss_map = np.squeeze(gen_gaussmap(config["boxsize"], config["sigma"]))

    pose_map = model.estimate(human, gauss_map)

    pose_map_resized = []
    joint_coords = []
    for joint_map in pose_map:
        # Resizes joint heatmaps
        joint_map_resized = cpm.map_resize(human.shape, joint_map)
        pose_map_resized.append(joint_map_resized)

        # Get resized coordinates of every joint
        joint_coord = model.get_coords(joint_map_resized, c, s,
                                       config["boxsize"])
        joint_coords.append(joint_coord)

    if viz:
        cpm.display_joints(sample, joint_coords)

    joint_coords = [[x, y] for y, x in joint_coords]

    return joint_coords


def estimate_tf_pose(sample, human, config, model, c, s, viz):
    """
    Estimate human pose given an input image.
    @param sample: np.array - original input image
    @param human: np.array - cropped human image
    @param config: dict - CPM settings
    @param model: pose estimator object
    @param c: np.array - human center
    @param s: int - human scale
    @param viz: bool - flag for joint visualization
    @return: np.array - joint coords
    """
    human = human / 256.0 - 0.5
    gauss_map = gen_gaussmap(config["boxsize"], config["sigma"])

    humans = np.dstack((np.squeeze(human), gauss_map))
    pose_estimator.humans = np.array([humans])

    pose_map = model.get_map()

    joint_coords = pose_estimator.get_joints(pose_map, c, s)

    if viz:
        im_drawn = model.draw_limbs(sample, joint_coords, config)
        plt.figure(), plt.imshow(im_drawn), plt.show()

    return joint_coords


if __name__ == '__main__':
    args = get_args()

    # Set framework
    base_path = ""
    if args["framework"] == "caffe":
        base_path = "../Estimator/Caffe/"
        from Estimator.Caffe import cpm
        from Estimator.Caffe.pose_estimator import PoseEstimator

    if args["framework"] == "tensorflow":
        base_path = "../Estimator/TensorFlow/"
        from Estimator.TensorFlow import cpm
        from Estimator.TensorFlow.pose_estimator import PoseEstimator

    # Read YAML file containing model info.
    data = None
    with open(base_path + "cpm.yml", "r") as stream:
        try:
            data = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Add base path to models & set device
    pose_estimator = None
    if args["framework"] == "caffe":
        for k, v in data["caffe_models"].items():
            data["caffe_models"][k] = base_path + v

        cpm.set_dev(data["GPU"])
        pose_estimator = load_caffe_model(data["caffe_models"])

    if args["framework"] == "tensorflow":
        for k, v in data["tf_models"].items():
            data["tf_models"][k] = base_path + v

        dev = cpm.set_dev(data["GPU"])
        pose_estimator = load_tf_model(data, dev)

    # Load .h5 file with subset annotations
    filename = args["subset"]
    f = h5py.File(filename, "r")
    anno = f["labels"]
    images = f["images"]

    mpii_folder = "../Datasets/MPII/images/"
    lsp_folder = "../Datasets/LSP/images/"

    fnames, centers, scales = (anno["fname"], anno["center"], anno["scale"])
    datasets = anno["dataset"]

    pred_idx = 0
    preds = np.zeros((2, 16, len(fnames)))

    iterator = zip(fnames, centers, scales, images, datasets)
    for fname, center, scale, im_human, dataset in iterator:
        # Read image
        im_folder = mpii_folder
        if dataset == "lsp":
            im_folder = lsp_folder

        im_path = im_folder + fname
        im = cv2.imread(im_path)

        # Image preprocessing
        # target_dist = 41. / 35.  # Mysterious parameter from original repo, maybe adjusted during training??
        # scale = target_dist / scale
        # center = np.uint(center * scale)
        #
        # im_human = crop_human(im, center, scale, data["boxsize"])

        # Pose estimation
        joints = None
        if args["framework"] == "caffe" and pose_estimator is not None:
            joints = estimate_caffe_pose(im, im_human, data, pose_estimator,
                                         center, scale, args["viz"])

        if args["framework"] == "tensorflow" and pose_estimator is not None:
            joints = estimate_tf_pose(im, im_human, data, pose_estimator,
                                      center, scale, args["viz"])

        if joints is not None:
            # Get joint coordinates ready for evaluation
            joints_standard = np.zeros((2, 16))
            joints_standard[:, 0] = joints[10]  # right ankle
            joints_standard[:, 1] = joints[9]   # right knee
            joints_standard[:, 2] = joints[8]   # right hip
            joints_standard[:, 3] = joints[11]  # left hip
            joints_standard[:, 4] = joints[12]  # left knee
            joints_standard[:, 5] = joints[13]  # left ankle
            # joints 6 & 7 are pelvis & thorax:
            #     - not calculated by CPMs
            #     - not taken into account when computing PCKh
            joints_standard[:, 8] = joints[1]   # upper neck
            joints_standard[:, 9] = joints[0]   # head top
            joints_standard[:, 10] = joints[4]  # right wrist
            joints_standard[:, 11] = joints[3]  # right elbow
            joints_standard[:, 12] = joints[2]  # right shoulder
            joints_standard[:, 13] = joints[5]  # left shoulder
            joints_standard[:, 14] = joints[6]  # left elbow
            joints_standard[:, 15] = joints[7]  # left wrist

            preds[:, :, pred_idx] = joints_standard

        pred_idx += 1

        print("%d/%d" % (pred_idx, len(fnames)))

    # Save results
    if not os.path.exists("preds"):
        os.makedirs("preds")

    date = time.strftime("%Y%m%d%H%M%S")
    subset = args["subset"].split("/")[-1].split(".")[0]
    pred_filename = "pred_%s_%s_%s_%s.h5" % (subset, args["framework"],
                                             data["boxsize"], date)

    pred_f = h5py.File("preds/" + pred_filename, "w")
    pred_f.create_dataset("preds", data=preds)

    pred_f.close()
