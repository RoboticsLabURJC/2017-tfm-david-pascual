#!/usr/bin/env python

"""
camera.py: Camera class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

from pprint import pprint
import sys
import threading
import traceback

import comm
import config
import numpy as np

import cv2


class Camera:
    def __init__(self):
        """
        Camera class gets images from live video and transform them
        in order to predict the digit in the image.
        """
        cfg = config.load(sys.argv[1])

        # starting comm
        jdrc = comm.init(cfg, "HumanPose")
        self.cam_rgb = False
        self.cam_depth = False

        self.lock = threading.Lock()

        # noinspection PyBroadException
        try:
            self.cam_rgb = jdrc.getCameraClient("HumanPose.CameraRGB")
            if self.cam_rgb.hasproxy():
                self.im_rgb = self.cam_rgb.getImage()
                self.im_height = self.im_rgb.height
                self.im_width = self.im_rgb.width
                print("RGB camera succesfully connected!")
        except:
            traceback.print_exc()
            exit()

        # noinspection PyBroadException
        try:
            self.cam_depth = jdrc.getCameraClient("HumanPose.CameraDEPTH")
            if self.cam_depth.hasproxy():
                self.im_depth = self.cam_depth.getImage()
                print("Depth camera succesfully connected!")
        except:
            # traceback.print_exc()
            print("Depth camera not found!")

    def update(self):
        """
        Updates Camera object.
        """
        if self.cam_rgb:
            self.lock.acquire()

            im_rgb = self.cam_rgb.getImage()

            self.im_rgb = im_rgb.data
            self.im_height = im_rgb.height
            self.im_width = im_rgb.width

            self.lock.release()

        if self.cam_depth:
            self.lock.acquire()

            im_depth = self.cam_depth.getImage()

            self.im_depth = im_depth.data

            self.lock.release()

    def get_image(self):
        """
        Get image from webcam.
        @return: np.array - Frame
        """
        im_rgb = np.zeros((self.im_height, self.im_width, 3))
        im_depth = np.zeros((self.im_height, self.im_width, 1))
        if self.cam_rgb:
            im_rgb = np.frombuffer(self.im_rgb.data, dtype=np.uint8)
            im_rgb.shape = self.im_height, self.im_width, 3

        if self.cam_depth:
            im_depth = np.frombuffer(self.im_depth.data, dtype=np.uint8)
            im_depth.shape = self.im_height, self.im_width, 3
            im_depth = im_depth[:, :, 0]

            cv2.imshow("rgbd", im_depth)
            cv2.waitKey(1)

        return im_rgb, im_depth
