#!/usr/bin/env python

"""
camera.py: Camera class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import sys
import threading
import traceback

import comm
import config
import numpy as np


class Camera:
    def __init__(self):
        """
        Camera class gets images from live video and transform them
        in order to predict the digit in the image.
        @param data: parsed YAML config. file
        """
        cfg = config.load(sys.argv[1])

        # starting comm
        jdrc = comm.init(cfg, "HumanPose")
        self.cam = jdrc.getCameraClient("HumanPose.Camera")

        self.lock = threading.Lock()

        # noinspection PyBroadException
        try:
            if self.cam.hasproxy():
                self.im = self.cam.getImage()
                self.im_height = self.im.height
                self.im_width = self.im.width
                print("Camera succesfully connected!")
        except:
            traceback.print_exc()
            exit()

    def update(self):
        """
        Updates Camera object.
        """
        if self.cam:
            self.lock.acquire()

            im = self.cam.getImage()

            self.im = im.data
            self.im_height = im.height
            self.im_width = im.width

            self.lock.release()

    def get_image(self):
        """
        Get image from webcam.
        @return: np.array - Frame
        """
        im = np.zeros((self.im_height, self.im_width, 3))
        if self.cam:
            im = np.frombuffer(self.im.data, dtype=np.uint8)
            im.shape = self.im_height, self.im_width, 3

        return im
