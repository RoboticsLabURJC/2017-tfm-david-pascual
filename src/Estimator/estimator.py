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
import pyquaternion as pq

from random import randint

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


def draw_3d_estimation(viz3d, im_depth, joints, limbs, colors, calib_data=None, draw_segments=False):
    if draw_segments:
        drawn_joints = []
        for l, (p, q) in enumerate(limbs):
            px, py = joints[p]
            qx, qy = joints[q]

            if px >= 0 and py >= 0 and qx >= 0 and qy >= 0:
                point_a = get_depth_point(joints[p], im_depth, calib_data)
                point_b = get_depth_point(joints[q], im_depth, calib_data)

                color = colors[l]
                viz3d.drawSegment(point_a, point_b, color)
                print(p, q, color)
                if p not in drawn_joints:
                    point = get_depth_point(joints[p], im_depth, calib_data)
                    viz3d.drawPoint(point, (255, 255, 255))
                    drawn_joints.append(p)
                if q not in drawn_joints:
                    point = get_depth_point(joints[q], im_depth, calib_data)
                    viz3d.drawPoint(point, (255, 255, 255))
                    drawn_joints.append(q)

    else:
        try:
            # TRONCO
            right_hip = get_depth_point(joints[8], im_depth, calib_data)
            left_hip = get_depth_point(joints[11], im_depth, calib_data)
            head = (get_depth_point(joints[1], im_depth, calib_data) + get_depth_point(joints[1], im_depth,
                                                                                       calib_data)) / 2

            tronco = (right_hip + left_hip) / 2
            tronco[0] += 75
            tronco[1] += 80
            tronco[2] -= 20

            yaw = math.atan2(tronco[1] - head[1], tronco[0] - head[0])
            roll = np.pi / 2
            pitch = math.atan2(math.sqrt((tronco[1] - head[1]) ** 2 + (tronco[0] - head[0]) ** 2),
                               tronco[0] - head[0]) + np.pi

            tronco_pitch = pq.Quaternion(axis=[1, 0, 0], degrees=0)
            tronco_roll = pq.Quaternion(axis=[0, 1, 0], degrees=np.pi)
            tronco_yaw = pq.Quaternion(axis=[0, 0, 1], degrees=np.pi)
            tronco_quat = tronco_roll

            viz3d.drawPose3d(0, tronco, tronco_quat, 0)

            # FEMUR DERECHO
            femur_dcho = right_hip
            femur_dcho[0] -= 20
            femur_dcho[1] += 60
            femur_dcho[2] += 10
            viz3d.drawPose3d(1, femur_dcho, tronco_quat, 0)

            # FEMUR IZQUIERDO
            femur_izqdo = left_hip
            femur_izqdo[0] += 50  # dcha / izqda
            femur_izqdo[1] += 80  # alante / atras
            femur_izqdo[2] += 10  # arriba / abajo
            viz3d.drawPose3d(2, femur_izqdo, tronco_quat, 0)

            # TIBIA DERECHA
            tibia_dcho = get_depth_point(joints[9], im_depth, calib_data)
            tibia_dcho[0] -= 20
            tibia_dcho[1] += 30
            tibia_dcho[2] += 20
            viz3d.drawPose3d(3, tibia_dcho, tronco_quat, 0)

            # TIBIA IZQUIERDA
            tibia_izqdo = get_depth_point(joints[12], im_depth, calib_data)
            tibia_izqdo[0] += 20
            tibia_izqdo[1] += 30
            tibia_izqdo[2] += 20
            viz3d.drawPose3d(4, tibia_izqdo, tronco_quat, 0)

            # PIE DERECHO
            pie_dcho = get_depth_point(joints[10], im_depth, calib_data)
            pie_dcho[0] -= 0
            pie_dcho[1] += 30
            pie_dcho[2] += 10
            viz3d.drawPose3d(5, pie_dcho, tronco_yaw * tronco_roll, 0)

            # PIE IZQUIERDO
            pie_izqdo = get_depth_point(joints[13], im_depth, calib_data)
            pie_izqdo[0] -= 0
            pie_izqdo[1] += 30
            pie_izqdo[2] += 10
            viz3d.drawPose3d(6, pie_izqdo, tronco_yaw * tronco_roll, 0)
        except IndexError:
            print("\tWARNING: Not all bones are ready yet!")


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
        all_humans, all_joints = self.estimate(im)

        colors = self.config["colors"]
        limbs = np.array(self.config["limbs"]).reshape((-1, 2)) - 1

        if self.cam_depth:
            im_depth = self.cam_depth.get_image().copy()
            for bbox, joints in zip(all_humans, all_joints):
                draw_3d_estimation(self.viz3d, im_depth, joints, limbs, colors, self.cam_depth.calib_data)
        else:
            if self.gui.display:
                for bbox, joints in zip(all_humans, all_joints):
                    im = draw_estimation(im, bbox, joints, limbs, colors)

            self.gui.im_pred = im.copy()
            self.gui.display = False
