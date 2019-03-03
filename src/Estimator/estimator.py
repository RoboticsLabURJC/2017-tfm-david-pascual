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

from random import randint

__author__ = "David Pascual Hernandez"
__date__ = "2018/27/05"


def draw_estimation(im, im_depth, bbox, joints, limbs, colors, stickwidth=6):
    if np.sum(im_depth):
        cv2.normalize(im_depth, im_depth, 0, 1, cv2.NORM_MINMAX)
        im_depth *= 255
        im_depth = np.dstack((im_depth, im_depth, im_depth)).astype(np.uint8)

    upper, lower = bbox
    cv2.rectangle(im, tuple(upper), tuple(lower), (0, 255, 0), 3)

    for i, (p, q) in enumerate(limbs):
        px, py = joints[p]
        qx, qy = joints[q]

        if px >= 0 and py >= 0 and qx >= 0 and qy >= 0:
            m_x = int(np.mean(np.array([px, qx])))
            m_y = int(np.mean(np.array([py, qy])))

            length = ((px - qx) ** 2. + (py - qy) ** 2.) ** 0.5
            angle = math.degrees(math.atan2(py - qy, px - qx))
            polygon = cv2.ellipse2Poly((m_x, m_y),
                                       (int(length / 2), stickwidth),
                                       int(angle), 0, 360, 1)
            cv2.fillConvexPoly(im, polygon, colors[i])
            if np.sum(im_depth):
                cv2.fillConvexPoly(im_depth, polygon, colors[i])

        if px >= 0 and py >= 0:
            cv2.circle(im, (px, py), 3, (0, 0, 0), -1)
            if np.sum(im_depth):
                cv2.circle(im_depth, (px, py), 3, (0, 0, 0), -1)
        if qx >= 0 and qy >= 0:
            cv2.circle(im, (qx, qy), 3, (0, 0, 0), -1)
            if np.sum(im_depth):
                cv2.circle(im_depth, (qx, qy), 3, (0, 0, 0), -1)

    if np.sum(im_depth):
        cv2.imshow("Depth map", im_depth)

    return im


def get_depth_point(point, depth, calib_data=None):
    """
    Get joints depth information.
    """
    height, width = depth.shape
    x, y = point

    search_area = 10  # search area for given coordinates to avoid noisy estimations

    if x < search_area:
        x = search_area
    if x > width - search_area:
        x = width - search_area
    if y < search_area:
        y = search_area
    if y > height - search_area:
        y = height - search_area

    depth_estimation = depth.copy()
    depth_estimation[np.where(depth_estimation == 0)] = np.nan
    z = np.nanmin(depth_estimation[y - search_area:y + search_area, x - search_area:x + search_area])

    y = height - y
    x = width - x
    if calib_data:
        x = (x - calib_data["cx"]) * z / calib_data["fx"]
        y = (y - calib_data["cy"]) * z / calib_data["fy"]

    return np.array((x, z, y))


def draw_3d_estimation(viz3d, im_depth, joints, limbs, colors, calib_data=None):
    drawn_joints = []
    for l, (p, q) in enumerate(limbs):
        px, py = joints[p]
        qx, qy = joints[q]

        if px >= 0 and py >= 0 and qx >= 0 and qy >= 0:
            point_a = get_depth_point(joints[p], im_depth, calib_data)
            point_b = get_depth_point(joints[q], im_depth, calib_data)

            color = colors[l]
            viz3d.drawSegment(point_a, point_b, color)

            if p not in drawn_joints:
                point = get_depth_point(joints[p], im_depth, calib_data)
                viz3d.drawPoint(point, (255, 255, 255))
                drawn_joints.append(p)
            if q not in drawn_joints:
                point = get_depth_point(joints[q], im_depth, calib_data)
                viz3d.drawPoint(point, (255, 255, 255))
                drawn_joints.append(q)

    # if len(real_world_joints):
    #     viz3d.drawPose3d(tronco_idx, real_world_joints[tronco_limb_idx], (0, 0, 0, 0), 0)



class Estimator:
    def __init__(self, cam, cam_depth, viz3d, gui, data):
        """
        Estimator class gets human pose estimations for a given image.
        @param cam: Camera object
        @param cam_depth: Depth camera object
        @param viz3d: Viz3D object
        @param gui: GUI object
        @param data: parsed YAML config. file
        """
        self.cam = cam
        self.cam_depth = cam_depth
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
        im = self.cam.get_image().copy()
        im_depth = np.zeros(im.shape[:2], np.float16)

        if self.cam_depth:
            im_depth = self.cam_depth.get_image().copy()

        all_humans, all_joints = self.estimate(im)

        colors = self.config["colors"]
        limbs = np.array(self.config["limbs"]).reshape((-1, 2)) - 1

        if self.gui.display:
            for bbox, joints in zip(all_humans, all_joints):
                if len(joints):
                    if self.cam_depth:
                        draw_3d_estimation(self.viz3d, im_depth, joints, limbs, colors, self.cam_depth.calib_data)
                    else:
                        draw_3d_estimation(self.viz3d, im_depth, joints, limbs, colors)
                    im = draw_estimation(im, im_depth, bbox, joints, limbs, colors)

        self.gui.im_pred = im.copy()
        self.gui.display = False
