#!/usr/bin/env python

"""
foo.py: bar.
"""

import math
import sys
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np

__author__ = "David Pascual Hernandez"
__date__ = "2018/27/05"


def draw_estimation(im, bbox, joints, limbs, colors, stickwidth=6):
    upper, lower = bbox
    cv2.rectangle(im, tuple(upper), tuple(lower), (0, 255, 0), 3)

    for i, (p, q) in enumerate(limbs):
        px, py = joints[p]
        qx, qy = joints[q]

        m_x = int(np.mean(np.array([px, qx])))
        m_y = int(np.mean(np.array([py, qy])))

        length = ((px - qx) ** 2. + (py - qy) ** 2.) ** 0.5
        angle = math.degrees(math.atan2(py - qy, px - qx))
        polygon = cv2.ellipse2Poly((m_x, m_y),
                                   (int(length / 2), stickwidth),
                                   int(angle), 0, 360, 1)
        cv2.fillConvexPoly(im, polygon, colors[i])

        cv2.circle(im, (px, py), 3, (0, 0, 0), -1)
        cv2.circle(im, (qx, qy), 3, (0, 0, 0), -1)

    return im


def get_depth_point(point, depth):
    """
    Get joints depth information.
    """
    y, x = point

    x = np.clip(int(x), 0, depth.shape[1] - 1)
    y = np.clip(int(depth.shape[0] - y), 0, depth.shape[0] - 1)

    z = depth[y, x]

    return np.array((x, z, y))


def draw_3d_estimation(viz3d, im_depth, joints, limbs, colors):
    limbs = np.array(limbs).reshape((-1, 2)) - 1
    for l, (p, q) in enumerate(limbs):
        point_a = get_depth_point(joints[p], im_depth)
        point_b = get_depth_point(joints[q], im_depth)

        color = colors[l]
        viz3d.drawSegment(point_a, point_b, color)

    for joint in joints[:-1]:
        point = get_depth_point(joint, im_depth)
        viz3d.drawPoint(point, (255, 255, 255))


class Estimator:
    def __init__(self, cam, viz3d, gui, data):
        """
        Estimator class gets human pose estimations for a given image.
        @param cam: Camera object
        @param viz3d: Viz3D object
        @param gui: GUI object
        @param data: parsed YAML config. file
        """
        self.cam = cam
        self.viz3d = viz3d
        self.gui = gui

        self.config = data
        sigma = self.config["sigma"]
        boxsize = self.config["boxsize"]
        human_framework = self.config["human_framework"]
        pose_framework = self.config["pose_framework"]

        available_human_fw = ["caffe", "tf", "naxvm"]
        available_pose_fw = ["caffe"]

        HumanDetector, PoseEstimator = (None, None)
        human_model, pose_model = (None, None)

        if human_framework == "caffe":
            from Human.human_caffe import HumanDetector
            human_model = self.config["caffe_models"]["human"]
        elif human_framework == "tf":
            from Human.human_tf import HumanDetector
            human_model = self.config["tf_models"]["human"]
        elif human_framework == "naxvm":
            from Human.human_naxvm import HumanDetector
            human_model = self.config["naxvm_models"]["human"]
        else:
            print("'%s' is not supported for human detection" % human_framework)
            print("Available frameworks: " + str(available_human_fw))
            exit()

        if pose_framework == "caffe":
            from Pose.pose_caffe import PoseEstimator
            pose_model = self.config["caffe_models"]["pose"]
        else:
            print("'%s' is not supported for pose detection" % pose_framework)
            print("Available frameworks: " + str(available_pose_fw))
            exit()

        self.hd = HumanDetector(human_model, boxsize)
        self.pe = PoseEstimator(pose_model, boxsize, sigma)

    def estimate(self, frame):
        """
        Estimate human pose.
        @param frame: np.array - Image, preferably with humans
        @return: np.array, np.array - joint coordinates & limbs drawn
        over original image
        """
        print("-" * 80)
        t = time.time()
        human_bboxes = self.hd.get_bboxes(frame)
        print("Human detection: %d ms" % int((time.time() - t) * 1000))

        all_joints = []
        for bbox in human_bboxes:
            t = time.time()
            joints = self.pe.get_coords(frame, bbox)
            all_joints.append(joints)
            print("\tPose estimation: %d ms" % int((time.time() - t) * 1000))

        return human_bboxes, np.array(all_joints)

    def update(self):
        """ Update estimator. """
        im, im_depth = self.cam.get_image()
        im, im_depth = (im.copy(), im_depth.copy())
        all_humans, all_joints = self.estimate(im)

        limbs = np.array(self.config["limbs"]).reshape((-1, 2)) - 1
        colors = self.config["colors"]
        for bbox, joints in zip(all_humans, all_joints):
            im = draw_estimation(im, bbox, joints, limbs, colors)

            if len(joints) and self.gui.display:
                draw_3d_estimation(self.viz3d, im_depth, joints, limbs, colors)

        self.gui.im_pred = im.copy()
        self.gui.display = False
