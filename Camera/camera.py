#!/usr/bin/env python

"""
camera.py: Camera class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import numpy as np
import sys
import threading
import traceback

import easyiceconfig as EasyIce
from jderobot import CameraPrx


class Camera:

    def __init__ (self):
        """
        Camera class gets images from live video and transform them
        in order to predict the digit in the image.
        """
        status = 0
        ic = None
        
        # Initialize Ice
        ic = EasyIce.initialize(sys.argv)

        properties = ic.getProperties()
        self.lock = threading.Lock()
    
        try:
            # Obtain a proxy for the camera
            obj = ic.propertyToProxy("Humanpose.Camera.Proxy")
            # Get first image and print its description.
            self.cam = CameraPrx.checkedCast(obj)
            if self.cam:
                self.im = self.cam.getImageData("RGB8")
                self.im_height = self.im.description.height
                self.im_width = self.im.description.width
                print(self.im.description)
            else: 
                print("Interface camera not connected")
                    
        except:
            traceback.print_exc()
            exit()
            status = 1


    def getImage(self):
        """
        Get image from webcam.
        :return: np.array - Frame
        """
        if self.cam:
            imageData = self.cam.getImageData("RGB8")
            imageData_h = imageData.description.height
            imageData_w = imageData.description.width
            image = np.zeros((imageData_h, imageData_w, 3), np.uint8)
            image = np.frombuffer(imageData.pixelData, dtype=np.uint8)
            image.shape = imageData_h, imageData_w, 3

        return image

    def update(self):
        """
        Update camera.
        """
        _ = self.getImage()
