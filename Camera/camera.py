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

import numpy as np
import easyiceconfig as EasyIce
from jderobot import CameraPrx


class Camera:
    def __init__(self):
        """
        Camera class gets images from live video and transform them
        in order to predict the digit in the image.
        """
        # Initialize Ice
        ic = EasyIce.initialize(sys.argv)

        self.lock = threading.Lock()

        try:
            # Obtain a proxy for the camera
            obj = ic.propertyToProxy("Humanpose.Camera.Proxy")
            # Get first image and print its description.
            self.cam = CameraPrx.checkedCast(obj)
            if self.cam:
                im = self.cam.getImageData("RGB8")
                self.im_height = im.description.height
                self.im_width = im.description.width
                print(im.description)
            else:
                print("Interface camera not connected")

        except:
            traceback.print_exc()
            exit()

    def get_image(self):
        """
        Get image from webcam.
        :return: np.array - Frame
        """
        im = np.zeros((self.im_width, self.im_height, 3))
        if self.cam:
            im_data = self.cam.getImageData("RGB8")
            im = np.frombuffer(im_data.pixelData, dtype=np.uint8)
            im.shape = self.im_height, self.im_width, 3

        return im
