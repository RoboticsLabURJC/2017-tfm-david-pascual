#!/usr/bin/env python

"""
testbench.py: Script for testing humanpose component performance.
"""

__author__ = "David Pascual Hernandez"
__date__ = "2017/02/03"

import argparse
import csv
import os
import time
import yaml

import cv2

from Estimator.Caffe import cpm as caffe_cpm
from Estimator.TensorFlow import cpm as tf_cpm


def get_args():
    """
    Get program arguments and parse them.
    :return: dict - arguments
    """
    ap = argparse.ArgumentParser(description="CPMs testbench")
    ap.add_argument("-v", "--video", type=str, required=True,
                    help="Input video")
    ap.add_argument("-c", "--config", type=str, required=False,
                    default="humanpose.yml", help="YAML config. file")
    ap.add_argument("-o", "--out", type=str, required=False,
                    default="testbench_results", help="Results folder")
    ap.add_argument("-f", "--framework", type=str, required=False,
                    default="", help="Framework to test")
    ap.add_argument("-b", "--boxsize", type=int, required=False,
                    default="", help="Boxsize used")
    ap.add_argument("-g", "--gpu_flag", type=int, required=False,
                    default="", help="GPU flag")

    return vars(ap.parse_args())


def parse_yaml(file):
    """
    Parse YAML config. file.
    @param file: str - YAML file path 
    @return: dict - parsed YAML file
    """
    data = None
    with open(file, "r") as stream:
        try:
            data = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return data


if __name__ == '__main__':
    # Read and set variables
    args = get_args()
    framework = args["framework"]
    boxsize = args["boxsize"]
    gpu = args["gpu_flag"]
    data = parse_yaml(args["config"])

    cap = cv2.VideoCapture(args["video"])

    settings = data["Settings"]
    if framework == "":
        framework = data["Framework"]
    if boxsize == "":
        boxsize = settings["boxsize"]
    if gpu == "":
        gpu = settings["GPU"]

    if gpu:
        dev = "GPU"
    else:
        dev = "CPU"

    # If output folder doesn't exist, then makedir
    if not os.path.exists(args["out"]):
        os.makedirs(args["out"])

    # Exclusive folder & .csv for this test
    date = time.strftime("%Y%m%d%H%M%S")
    out_folder = args["out"] + "/%s-%s-%s-%s/" % (framework, boxsize, dev, date)
    os.makedirs(out_folder)

    # Load and set the models
    tf_config, deploy_models, model_paths = [None] * 3
    if framework == "caffe":
        models = caffe_cpm.load_model(settings["caffe_models"])
        caffe_cpm.set_dev(gpu)

    elif framework == "tensorflow":
        im_shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
        pose_shape = (boxsize, boxsize, 15)

        tf_config = tf_cpm.set_dev(gpu)
        models = tf_cpm.load_model(settings, im_shape, pose_shape, tf_config,
                                   boxsize)

    else:
        print(framework, " framework is not supported")
        print("Available frameworks: 'caffe', 'tensorflow'")
        exit()

    n_frame = 0
    im_pred = None
    full_pred_t = []
    pred_times = []
    n_humans = []

    # Read input video, display prediction & save results
    real_start_t = time.time()
    while cap.isOpened():
        start_t = time.time()
        ret, frame = cap.read()

        if ret:
            # Human pose prediction
            pred_t = ["", ""]
            im_predicted, pose_coords = [None, None]
            if framework == "caffe":
                im_pred, pose_coords, pred_t = caffe_cpm.predict(frame,
                                                                 settings,
                                                                 models,
                                                                 boxsize,
                                                                 viz=False)

            if framework == "tensorflow":
                im_pred, pose_coords, pred_t = tf_cpm.predict(frame, models,
                                                              settings,
                                                              viz=False)

            pred_times.append(pred_t)
            n_humans.append(len(pose_coords))

            # Store predicted images
            if type(im_pred) != type(None):
                cv2.imshow('Prediction', im_pred)
                cv2.imwrite(out_folder + "%04d" % n_frame + ".png", im_pred)
            else:
                print("No prediction !! ")

            n_frame += 1

        else:
            break

        full_pred_t.append(1000 * (time.time() - start_t))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Store performance data in .csv file
    total_t = time.time() - real_start_t

    with open(out_folder + "pred_times-%06d.csv" % (total_t), "wb")\
            as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["n Humans", "Human(ms)", "Pose(ms)", "Total"])

        # Prediction times
        for n, (human_t, pose_t), full_t in zip(n_humans, pred_times,
                                                full_pred_t):
            writer.writerow([n, str(human_t), str(pose_t), str(full_t)])

    cap.release()
    cv2.destroyAllWindows()
