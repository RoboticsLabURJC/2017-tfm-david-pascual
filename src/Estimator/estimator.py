#!/usr/bin/env python

"""
foo.py: bar.
"""

import math
import time

import cv2
import numpy as np

from kfilter import KFilter3D
import utils

__author__ = "David Pascual Hernandez"
__date__ = "2018/27/05"


def get_depth_point(point, depth, calib_data=None, kfilter=None):
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

    if kfilter is not None:
        kfilter.update()

    return x, z, y


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
        self.idx = 0

        self.cam = cam
        self.cam_depth = cam_depth
        self.viz3d = viz3d
        self.gui = gui

        self.config = data
        boxsize = self.config["boxsize"]
        human_framework = self.config["human_framework"]
        pose_framework = self.config["pose_framework"]

        available_human_fw = ["cpm_caffe", "cpm_tf", "naxvm"]
        available_pose_fw = ["cpm_caffe"]

        HumanDetector, PoseEstimator = (None, None)
        human_model = self.config["human_models"][human_framework]
        pose_model = self.config["pose_models"][pose_framework]

        if human_framework == "cpm_caffe":
            from Human.human_cpm_caffe import HumanDetector
        elif human_framework == "cpm_tf":
            from Human.human_cpm_tf import HumanDetector
        elif human_framework == "naxvm":
            from Human.human_naxvm import HumanDetector
        else:
            print("'%s' is not supported for human detection" % human_framework)
            print("Available frameworks: " + str(available_human_fw))
            exit()
        self.hd = HumanDetector(human_model, boxsize)

        if pose_framework == "cpm_caffe":
            from Pose.pose_cpm import PoseCPM
            sigma = self.config["cpm_config"]["sigma"]
            self.pe = PoseCPM(pose_model, boxsize, sigma)
        elif pose_framework == "stacked":
            from Pose.pose_stacked import PoseStacked
            self.pe = PoseStacked(pose_model, boxsize)
        else:
            print("'%s' is not supported for pose detection" % pose_framework)
            print("Available frameworks: " + str(available_pose_fw))
            exit()
        self.pos3d_kfilters = [KFilter3D() for idx in self.config["limbs"]]

    def estimate(self, frame):
        """
        Estimate human pose.
        @param frame: np.array - Image, preferably with humans
        @return: np.array, np.array - joint coordinates & limbs drawn
        over original image
        """
        # print("-" * 80)
        t = time.time()
        human_bboxes = self.hd.get_bboxes(frame)[:1]
        print("Human detection: %d ms" % int((time.time() - t) * 1000))

        all_joints = []
        for bbox in human_bboxes:
            t = time.time()
            joints = self.pe.get_coords(frame, bbox)
            all_joints.append(joints)
            print("\tPose estimation: %d ms" % int((time.time() - t) * 1000))

        return human_bboxes, np.array(all_joints)

    def draw_3d_segments(self, im_depth, joints, limbs, colors, filter=True):
        def get_joint_depth(idx):
            pos = get_depth_point(joints[0], im_depth, self.cam_depth.calib_data)
            if filter:
                pos = self.pos3d_kfilters[idx].update_filter(pos)
            return np.array(pos)

        drawn_joints = []
        for l, (p, q) in enumerate(limbs):
            px, py = joints[p]
            qx, qy = joints[q]

            if px >= 0 and py >= 0 and qx >= 0 and qy >= 0:
                point_a = get_joint_depth(p)
                point_b = get_joint_depth(q)

                color = colors[l]
                self.viz3d.drawSegment(point_a, point_b, color)
                if p not in drawn_joints:
                    point = get_joint_depth(p)
                    self.viz3d.drawPoint(point, (255, 255, 255))
                    drawn_joints.append(p)
                if q not in drawn_joints:
                    point = get_joint_depth(q)
                    self.viz3d.drawPoint(point, (255, 255, 255))
                    drawn_joints.append(q)

    def draw_3d_skeleton(self, im_depth, joints, filter=True):
        def get_joint_depth(idx):
            print("Joint %d" % idx)
            print("pos2d", joints[idx])
            if np.count_nonzero(np.where(joints[idx] == -1)):
                pos = (np.nan, np.nan, np.nan)
            else:
                pos = get_depth_point(joints[idx], im_depth, self.cam_depth.calib_data)
            print("pos3d", pos)
            if filter:
                pos = self.pos3d_kfilters[idx].update_filter(pos)
            print("posFiltered", pos)
            return np.array(pos)

        head_top = get_joint_depth(0)
        upper_neck = get_joint_depth(1)
        shoulder_right = get_joint_depth(2)
        elbow_right = get_joint_depth(3)
        wrist_right = get_joint_depth(4)
        shoulder_left = get_joint_depth(5)
        elbow_left = get_joint_depth(6)
        wrist_left = get_joint_depth(7)
        hip_right = get_joint_depth(8)
        knee_right = get_joint_depth(9)
        ankle_right = get_joint_depth(10)
        hip_left = get_joint_depth(11)
        knee_left = get_joint_depth(12)
        ankle_left = get_joint_depth(13)

        front_votes = 0
        back_votes = 0
        if shoulder_right[0] > shoulder_left[0]:
            back_votes +=1
        else:
            front_votes += 1
        if elbow_right[0] > elbow_left[0]:
            back_votes +=1
        else:
            front_votes += 1
        if wrist_right[0] > wrist_left[0]:
            back_votes +=1
        else:
            front_votes += 1
        if hip_right[0] > hip_left[0]:
            back_votes +=1
        else:
            front_votes += 1
        if knee_right[0] > knee_left[0]:
            back_votes +=1
        else:
            front_votes += 1
        if knee_right[0] > knee_left[0]:
            back_votes +=1
        else:
            front_votes += 1
        if ankle_right[0] > ankle_left[0]:
            back_votes +=1
        else:
            front_votes += 1

        orientation = "front"
        if back_votes > front_votes:
            orientation = "back"

        # Head
        head_quaternion = utils.get_quaternion(head_top, upper_neck, orientation)
        self.viz3d.drawPose3d(0, upper_neck, head_quaternion, 0)

        # Torso
        hip = np.mean((hip_right, hip_left), axis=0)
        torso_quaternion = utils.get_quaternion(upper_neck, hip, orientation)
        self.viz3d.drawPose3d(1, hip, torso_quaternion, 0)

        # Right arm
        arm_right_quaternion = utils.get_quaternion(shoulder_right, elbow_right, orientation)
        self.viz3d.drawPose3d(2, elbow_right, arm_right_quaternion, 0)

        # Left arm
        arm_left_quaternion = utils.get_quaternion(shoulder_left, elbow_left, orientation)
        self.viz3d.drawPose3d(3, elbow_left, arm_left_quaternion, 0)

        # Right forearm
        forearm_right_quaternion = utils.get_quaternion(elbow_right, wrist_right, orientation)
        self.viz3d.drawPose3d(4, wrist_right, forearm_right_quaternion, 0)

        # Left forearm
        forearm_left_quaternion = utils.get_quaternion(elbow_left, wrist_left, orientation)
        self.viz3d.drawPose3d(5, wrist_left, forearm_left_quaternion, 0)

        # Right thigh
        arm_right_quaternion = utils.get_quaternion(hip_right, knee_right, orientation)
        self.viz3d.drawPose3d(6, knee_right, arm_right_quaternion, 0)

        # Left thigh
        arm_left_quaternion = utils.get_quaternion(hip_left, knee_left, orientation)
        self.viz3d.drawPose3d(7, knee_left, arm_left_quaternion, 0)

        # Right leg
        forearm_right_quaternion = utils.get_quaternion(knee_right, ankle_right, orientation)
        self.viz3d.drawPose3d(8, ankle_right, forearm_right_quaternion, 0)

        # Left leg
        forearm_left_quaternion = utils.get_quaternion(knee_left, ankle_left, orientation)
        self.viz3d.drawPose3d(9, ankle_left, forearm_left_quaternion, 0)



    def update(self):
        """ Update estimator. """
        im = self.cam.get_image().copy()
        all_humans, all_joints = self.estimate(im)

        colors = self.config["colors"]
        limbs = np.array(self.config["limbs"]).reshape((-1, 2)) - 1

        if self.cam_depth:
            im_depth = self.cam_depth.get_image().copy()
            for bbox, joints in zip(all_humans, all_joints):
                try:
                    self.viz3d.drawPoint((0, 0, 0), (0, 0, 0))
                    self.viz3d.drawSegment((0, 0, 0), (200, 0, 0), (255, 0, 0))
                    self.viz3d.drawSegment((0, 0, 0), (0, 200, 0), (0, 255, 0))
                    self.viz3d.drawSegment((0, 0, 0), (0, 0, 200), (0, 0, 255))

                    self.draw_3d_skeleton(im_depth, joints, filter=True)
                except IndexError:
                    print("\tWARNING: Not all bones are ready yet!")
        else:
            # if self.gui.display:
            for bbox, joints in zip(all_humans, all_joints):
                im = draw_estimation(im, bbox, joints, limbs, colors)
                # cv2.imwrite("results/im_pose_%02d.png" % self.idx, im)
                self.idx += 1

            self.gui.im_pred = im.copy()
            self.gui.display = False
