#!/usr/bin/env python

"""
compute_pck.py: Compute PCK metric.

Based on: http://human-pose.mpi-inf.mpg.de/#evaluation
"""

import argparse
import math
from matplotlib import pyplot as plt

import h5py
import numpy as np

__author__ = "David Pascual Hernandez"
__date__ = "2018/04/09"


def get_args():
    """
    Get program arguments and parse them.
    :return: dict - arguments
    """
    ap = argparse.ArgumentParser(description="Compute PCK metric")
    ap.add_argument("-p", "--preds", type=str, required=True,
                    help="Predictions file")
    ap.add_argument("-s", "--subset", type=str, required=True,
                    help="Subset groundtruth")
    return vars(ap.parse_args())


def get_dist(joint_pred, joint_real, hsize):
    x_pred, y_pred = joint_pred
    x_real, y_real = joint_real

    if x_real and y_real:
        dist = math.sqrt((x_pred - x_real) ** 2 + (y_pred - y_real) ** 2)
        dist /= hsize
    else:
        dist = np.nan

    return dist


if __name__ == "__main__":
    args = get_args()
    preds_path = args["preds"]
    subset = args["subset"]

    preds = h5py.File(preds_path, "r")["preds"]

    anno = h5py.File(subset, "r")["labels"]
    fnames = anno["fname"]
    gt_joints_all = anno["joints"]
    hsizes = anno["headsize"]

    results = {"ankle": {"idx": [0, 5], "dists": [], "pct": 0, "curve": 0},
               "knee": {"idx": [1, 4], "dists": [], "pct": 0, "curve": 0},
               "hip": {"idx": [2, 3], "dists": [], "pct": 0, "curve": 0},
               "head": {"idx": [8, 9], "dists": [], "pct": 0, "curve": 0},
               "wrist": {"idx": [10, 15], "dists": [], "pct": 0, "curve": 0},
               "elbow": {"idx": [11, 14], "dists": [], "pct": 0, "curve": 0},
               "shoulder": {"idx": [12, 13], "dists": [], "pct": 0, "curve": 0},
               "total": {"idx": range(15), "dists": [], "pct": 0, "curve": 0}}

    dists = np.zeros((preds.shape[2], preds.shape[1]))
    iterator = enumerate(zip(fnames, gt_joints_all, hsizes))
    for i, (fname, gt_joints, hsize) in iterator:
        for j, gt_joint in enumerate(gt_joints):
            if j in [6, 7]:
                gt_joint = (0, 0)
            pred_joint = preds[:, j, i]
            dists[i, j] = get_dist(pred_joint, gt_joint, hsize)

    max_th = 0.5
    step = 0.01
    thresholds = np.arange(0, max_th, step)

    print("\n-------- PCKh @ %.1f --------\n" % max_th)

    plt.figure()
    subset_id = subset.split("/")[-1].split(".")[0]
    preds_id = subset.split("/")[-1].split(".")[0]
    plt.suptitle("PCKh - Subset: %s; Preds: %s" % (subset_id, preds_id))
    for i, k in enumerate(results.keys()):
        idx = results[k]["idx"]

        k_dists = np.ravel(dists[:, idx])

        pck_joint = []
        for th in thresholds:
            n_correct = len([d for d in k_dists if d < th])
            n_total = len([d for d in k_dists if not np.isnan(d)])

            pck_joint.append(100 * float(n_correct) / n_total)

        results[k]["dists"] = k_dists
        results[k]["pct"] = pck_joint[-1]
        results[k]["curve"] = pck_joint

        print("\t%s: %.2f%%" % (k, pck_joint[-1]))

        plt.subplot(2, 4, i + 1)
        plt.plot(thresholds, pck_joint)

        plt.title("PCKh %s (%.2f%% @ %.1f)" % (k, pck_joint[-1], max_th))

        plt.xlim((0, max_th))
        plt.ylim((0, 100)), plt.ylabel("%")

    plt.show()
