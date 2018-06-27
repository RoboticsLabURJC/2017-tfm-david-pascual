#!/usr/bin/env python

"""
viz_testbench.py: Script for 'testbench.py' results visualization.
"""

__author__ = "David Pascual Hernandez"
__date__ = "2017/02/06"

import glob
import os
import sys
from matplotlib import pyplot as plt

import numpy as np

if __name__ == '__main__':
    path_in = sys.argv[1]

    """
    READ STORED DATA
    """
    path_subs = [tup[0] for tup in os.walk(path_in)][1:]
    name_subs = [path_sub.split("/")[-1] for path_sub in path_subs]

    data = []
    for path, name in zip(path_subs, name_subs):
        framework, boxsize, dev = name.split("-")[:-1]

        single_data = {}
        single_data["framework"] = framework
        single_data["boxsize"] = boxsize
        single_data["dev"] = dev

        csv_path = glob.glob(path + "/*.csv")[0]
        times = np.genfromtxt(csv_path, delimiter=',', skip_header=1)

        single_data["n_humans"] = []
        single_data["human_t"] = []
        single_data["pose_t"] = []
        single_data["total_t"] = []
        for n, ht, pt, t in times:
            single_data["n_humans"].append(n)
            single_data["human_t"].append(ht)
            single_data["pose_t"].append(pt)
            single_data["total_t"].append(t)

        data.append(single_data)

    """
    VISUALIZATION
    """
    # n_frame vs mean total time (tf gpu & caffe gpu)
    x_tf, y_tf, y_ca, x_ca = [None] * 4
    for single in data:
        if single["boxsize"] == "192" and single["dev"] == "GPU":
            if single["framework"] == "tensorflow":
                y_tf = single["total_t"]
                x_tf = range(len(y_tf))

            if single["framework"] == "caffe":
                y_ca = single["total_t"]
                x_ca = range(len(y_ca))

    plt.figure()
    plt.plot(x_tf, y_tf, label="TF GPU - Boxsize=192")
    plt.plot(x_ca, y_ca, label="Caffe GPU - Boxsize=192")
    plt.ylabel("Full inference time (ms)")
    plt.xlabel("Frame")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    # boxsize vs mean times (total, pose & human for TF/Caffe/GPU/CPU)
    tf_cpu_total = [None] * 4
    tf_gpu_total = [None] * 4
    ca_cpu_total = [None] * 4
    ca_gpu_total = [None] * 4

    tf_cpu_human = [None] * 4
    tf_gpu_human = [None] * 4
    ca_cpu_human = [None] * 4
    ca_gpu_human = [None] * 4

    tf_cpu_pose = [None] * 4
    tf_gpu_pose = [None] * 4
    ca_cpu_pose = [None] * 4
    ca_gpu_pose = [None] * 4

    for single in data:
        if single["framework"] == "tensorflow":
            if single["dev"] == "CPU":
                if single["boxsize"] == "96":
                    tf_cpu_total[0] = np.nanmean(single["total_t"])
                    tf_cpu_human[0] = np.nanmean(single["human_t"])
                    tf_cpu_pose[0] = np.nanmean(single["pose_t"])
                if single["boxsize"] == "128":
                    tf_cpu_total[1] = np.nanmean(single["total_t"])
                    tf_cpu_human[1] = np.nanmean(single["human_t"])
                    tf_cpu_pose[1] = np.nanmean(single["pose_t"])
                if single["boxsize"] == "192":
                    tf_cpu_total[2] = np.nanmean(single["total_t"])
                    tf_cpu_human[2] = np.nanmean(single["human_t"])
                    tf_cpu_pose[2] = np.nanmean(single["pose_t"])
                if single["boxsize"] == "320":
                    tf_cpu_total[3] = np.nanmean(single["total_t"])
                    tf_cpu_human[3] = np.nanmean(single["human_t"])
                    tf_cpu_pose[3] = np.nanmean(single["pose_t"])

            if single["dev"] == "GPU":
                if single["boxsize"] == "96":
                    tf_gpu_total[0] = np.nanmean(single["total_t"])
                    tf_gpu_human[0] = np.nanmean(single["human_t"])
                    tf_gpu_pose[0] = np.nanmean(single["pose_t"])
                if single["boxsize"] == "128":
                    tf_gpu_total[1] = np.nanmean(single["total_t"])
                    tf_gpu_human[1] = np.nanmean(single["human_t"])
                    tf_gpu_pose[1] = np.nanmean(single["pose_t"])
                if single["boxsize"] == "192":
                    tf_gpu_total[2] = np.nanmean(single["total_t"])
                    tf_gpu_human[2] = np.nanmean(single["human_t"])
                    tf_gpu_pose[2] = np.nanmean(single["pose_t"])
                if single["boxsize"] == "320":
                    tf_gpu_total[3] = np.nanmean(single["total_t"])
                    tf_gpu_human[3] = np.nanmean(single["human_t"])
                    tf_gpu_pose[3] = np.nanmean(single["pose_t"])

        if single["framework"] == "caffe":
            if single["dev"] == "CPU":
                if single["boxsize"] == "96":
                    ca_cpu_total[0] = np.nanmean(single["total_t"])
                    ca_cpu_human[0] = np.nanmean(single["human_t"])
                    ca_cpu_pose[0] = np.nanmean(single["pose_t"])
                if single["boxsize"] == "128":
                    ca_cpu_total[1] = np.nanmean(single["total_t"])
                    ca_cpu_human[1] = np.nanmean(single["human_t"])
                    ca_cpu_pose[1] = np.nanmean(single["pose_t"])
                if single["boxsize"] == "192":
                    ca_cpu_total[2] = np.nanmean(single["total_t"])
                    ca_cpu_human[2] = np.nanmean(single["human_t"])
                    ca_cpu_pose[2] = np.nanmean(single["pose_t"])
                if single["boxsize"] == "320":
                    ca_cpu_total[3] = np.nanmean(single["total_t"])
                    ca_cpu_human[3] = np.nanmean(single["human_t"])
                    ca_cpu_pose[3] = np.nanmean(single["pose_t"])

            if single["dev"] == "GPU":
                if single["boxsize"] == "96":
                    ca_gpu_total[0] = np.nanmean(single["total_t"])
                    ca_gpu_human[0] = np.nanmean(single["human_t"])
                    ca_gpu_pose[0] = np.nanmean(single["pose_t"])
                if single["boxsize"] == "128":
                    ca_gpu_total[1] = np.nanmean(single["total_t"])
                    ca_gpu_human[1] = np.nanmean(single["human_t"])
                    ca_gpu_pose[1] = np.nanmean(single["pose_t"])
                if single["boxsize"] == "192":
                    ca_gpu_total[2] = np.nanmean(single["total_t"])
                    ca_gpu_human[2] = np.nanmean(single["human_t"])
                    ca_gpu_pose[2] = np.nanmean(single["pose_t"])
                if single["boxsize"] == "320":
                    ca_gpu_total[3] = np.nanmean(single["total_t"])
                    ca_gpu_human[3] = np.nanmean(single["human_t"])
                    ca_gpu_pose[3] = np.nanmean(single["pose_t"])

    plt.figure()
    boxsizes = [96, 128, 192, 320]

    plt.subplot(131)
    plt.plot(boxsizes, tf_cpu_human, color="red", label="TF CPU")
    plt.plot(boxsizes, tf_gpu_human, color="orange", label="TF GPU")
    plt.plot(boxsizes, ca_cpu_human, color="blue", label="Caffe CPU")
    plt.plot(boxsizes, ca_gpu_human, color="cyan", label="Caffe GPU")
    plt.plot(boxsizes, tf_cpu_human, "o", color="red")
    plt.plot(boxsizes, tf_gpu_human, "o", color="orange")
    plt.plot(boxsizes, ca_cpu_human, "o", color="blue")
    plt.plot(boxsizes, ca_gpu_human, "o", color="cyan")
    plt.ylabel("Human detection time (ms)")
    plt.xlabel("Boxsize (px)")

    plt.subplot(132)
    plt.plot(boxsizes, tf_cpu_pose, color="red", label="TF CPU")
    plt.plot(boxsizes, tf_gpu_pose, color="orange", label="TF GPU")
    plt.plot(boxsizes, ca_cpu_pose, color="blue", label="Caffe CPU")
    plt.plot(boxsizes, ca_gpu_pose, color="cyan", label="Caffe GPU")
    plt.plot(boxsizes, tf_cpu_pose, "o", color="red")
    plt.plot(boxsizes, tf_gpu_pose, "o", color="orange")
    plt.plot(boxsizes, ca_cpu_pose, "o", color="blue")
    plt.plot(boxsizes, ca_gpu_pose, "o", color="cyan")
    plt.ylabel("Pose estimation time (ms)")
    plt.xlabel("Boxsize (px)")

    plt.subplot(133)
    plt.plot(boxsizes, tf_cpu_total, color="red", label="TF CPU")
    plt.plot(boxsizes, tf_gpu_total, color="orange", label="TF GPU")
    plt.plot(boxsizes, ca_cpu_total, color="blue", label="Caffe CPU")
    plt.plot(boxsizes, ca_gpu_total, color="cyan", label="Caffe GPU")
    plt.plot(boxsizes, tf_cpu_total, "o", color="red")
    plt.plot(boxsizes, tf_gpu_total, "o",  color="orange")
    plt.plot(boxsizes, ca_cpu_total, "o", color="blue")
    plt.plot(boxsizes, ca_gpu_total, "o", color="cyan")
    plt.ylabel("Full inference time (ms)")
    plt.xlabel("Boxsize (px)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    print("TF CPU:")
    for i, boxsize in enumerate(boxsizes):
        print("\t Boxsize %03d -> Human: %05dms; Pose: %05dms; Total: %05dms"
              % (boxsize, tf_cpu_human[i], tf_cpu_pose[i], tf_cpu_total[i]))

    print("TF GPU:")
    for i, boxsize in enumerate(boxsizes):
        print("\t Boxsize %03d -> Human: %05dms; Pose: %05dms; Total: %05dms"
              % (boxsize, tf_gpu_human[i], tf_gpu_pose[i], tf_gpu_total[i]))

    print("Caffe CPU:")
    for i, boxsize in enumerate(boxsizes):
        print("\t Boxsize %03d -> Human: %05dms; Pose: %05dms; Total: %05dms"
              % (boxsize, ca_cpu_human[i], ca_cpu_pose[i], ca_cpu_total[i]))

    print("Caffe GPU:")
    for i, boxsize in enumerate(boxsizes):
        print("\t Boxsize %03d -> Human: %05dms; Pose: %05dms; Total: %05dms"
              % (boxsize, ca_gpu_human[i], ca_gpu_pose[i], ca_gpu_total[i]))

    plt.show()

    # n humans vs mean pose time
    tf_gpu = [None] * 3
    ca_gpu = [None] * 3
    for single in data:
        if single["boxsize"] == "192" and single["dev"] == "GPU":
            if single["framework"] == "tensorflow":
                n_humans = np.array(single["n_humans"])
                pose_t = np.array(single["pose_t"])

                tf_gpu[0] = np.mean(pose_t[n_humans == 1])
                tf_gpu[1] = np.mean(pose_t[n_humans == 2])
                tf_gpu[2] = np.mean(pose_t[n_humans == 3])

            if single["framework"] == "caffe":
                n_humans = np.array(single["n_humans"])
                pose_t = np.array(single["pose_t"])

                ca_gpu[0] = np.mean(pose_t[n_humans == 1])
                ca_gpu[1] = np.mean(pose_t[n_humans == 2])
                ca_gpu[2] = np.mean(pose_t[n_humans == 3])

    plt.figure()
    n_humans = [1, 2, 3]

    plt.plot(n_humans, tf_gpu, color="r", label="TF GPU - Boxsize=192")
    plt.plot(n_humans, ca_gpu, color="b", label="Caffe GPU - Boxsize=192")
    plt.plot(n_humans, tf_gpu, "o", color="r")
    plt.plot(n_humans, ca_gpu, "o", color="b")
    plt.ylabel("Pose estimation time (ms)")
    plt.xlabel("Number of humans")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()