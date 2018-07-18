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

        if px >= 0 and py >= 0 and qx >= 0 and qy >= 0:
            m_x = int(np.mean(np.array([px, qx])))
            m_y = int(np.mean(np.array([py, qy])))

            length = ((px - qx) ** 2. + (py - qy) ** 2.) ** 0.5
            angle = math.degrees(math.atan2(py - qy, px - qx))
            polygon = cv2.ellipse2Poly((m_x, m_y),
                                       (int(length / 2), stickwidth),
                                       int(angle), 0, 360, 1)
            cv2.fillConvexPoly(im, polygon, colors[i])

        if px >= 0 and py >= 0:
            cv2.circle(im, (px, py), 3, (0, 0, 0), -1)
        if qx >= 0 and qy >= 0:
            cv2.circle(im, (qx, qy), 3, (0, 0, 0), -1)

    return im


def get_depth_point(point, depth):
    """
    Get joints depth information.
    """
    x, y = point

    x = np.clip(int(x), 0, depth.shape[1] - 1)
    y = np.clip(int(y), 0, depth.shape[0] - 1)

    # Median of the neighbors to find noiseless depth estimation
    if y - 5 >= 0:
        lower = y - 5
    else:
        lower = 0

    if y + 5 < depth.shape[0]:
        upper = y + 5
    else:
        upper = depth.shape[0] - 1

    if x - 5 >= 0:
        lefter = x - 5
    else:
        lefter = 0

    if x + 5 < depth.shape[1]:
        righter = x + 5
    else:
        righter = depth.shape[1] - 1
    z = np.nanmedian(depth[lower:upper, lefter:righter])

    y = depth.shape[0] - y

    return np.array((x, -int(z), y))


def draw_3d_estimation(viz3d, im_depth, joints, limbs, colors):
    for l, (p, q) in enumerate(limbs):
        px, py = joints[p]
        qx, qy = joints[q]

        if px >= 0 and py >= 0 and qx >= 0 and qy >= 0:
            point_a = get_depth_point(joints[p], im_depth, median=True)
            point_b = get_depth_point(joints[q], im_depth, median=True)

            color = colors[l]
            viz3d.drawSegment(point_a, point_b, color)

    for joint in joints[:-1]:
        x, y = joint
        if x >= 0 and y >= 0:
            point = get_depth_point(joint, im_depth)
            viz3d.drawPoint(point, (255, 255, 255))


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

        im_depth = []
        if self.cam_depth:
            im_depth = self.cam_depth.get_image().copy()

            # # We must get rid of this conversion!
            # im_depth_disp = im_depth.astype(float)
            # im_depth_disp = (im_depth_disp / float(4096)) * 255
            # im_depth_disp =  im_depth_disp.astype(np.uint8)
            # cv2.imshow("Depth image", im_depth_disp)
            # cv2.waitKey(1)
            #
            # self.viz3d.drawPoint(np.array((0, 0, 0)), (0, 0, 0))
            # self.viz3d.drawPoint(np.array((640, 0, 0)), (255, 0, 0))
            # self.viz3d.drawPoint(np.array((640, 4096, 0)), (255, 255, 0))
            # self.viz3d.drawPoint(np.array((640, 0, 480)), (255, 0, 255))
            # self.viz3d.drawPoint(np.array((640, 4096, 480)), (255, 255, 255))
            # self.viz3d.drawPoint(np.array((0, 4096, 0)), (0, 255, 0))
            # self.viz3d.drawPoint(np.array((0, 4096, 480)), (0, 255, 255))
            # self.viz3d.drawPoint(np.array((0, 0, 480)), (0, 0, 255))
            # for i in range(140, 460, 5):  # im_depth.shape[1]):
            #     for j in range(100, 400, 5):  # im_depth.shape[0]):
            #         point = get_depth_point((i, j), im_depth)
            #         print(i,j, im_depth[j, i], point)
            #         self.viz3d.drawPoint(point, tuple(im[j, i]/255.))

        all_humans, all_joints = self.estimate(im)

        colors = self.config["colors"]
        limbs = np.array(self.config["limbs"]).reshape((-1, 2)) - 1
        for bbox, joints in zip(all_humans, all_joints):
            im = draw_estimation(im, bbox, joints, limbs, colors)

            if len(joints) and len(im_depth) and self.gui.display:
                draw_3d_estimation(self.viz3d, im_depth, joints, limbs, colors)

        self.gui.im_pred = im.copy()
        self.gui.display = False
