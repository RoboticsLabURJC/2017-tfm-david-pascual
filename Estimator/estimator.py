#!/usr/bin/env python

"""
estimator.py: Estimator class.

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

from Caffe import caffe_cpm as cpm


class Estimator:

    def __init__(self):
        """
        Estimator class gets human pose estimations for a given image.
        """
        print("\nLoading Caffe models...")
        self.model, self.deploy_models = cpm.load_model()
        print("loaded\n")

        self.busy = 0


    def estimate(self, im):
        """
        Estimate human pose.
        :param im: np.array - Image, preferably with humans
        :return: np.array, np.array - joint coordinates & limbs drawn
        over original image
        """
        pose_coords, im_predicted = cpm.predict(self.model,
                                                self.deploy_models,
                                                im, viz=False)

        return pose_coords, im_predicted


    def setGUI(self, gui):
        """
        Set GUI.
        :param gui: GUI object
        """
        self.gui = gui


    def setCamera(self, cam):
        """
        Set camera.
        :param cam: Camera object
        """
        self.cam = cam


    def update(self):
        """
        Update estimator.
        """
        print("\nme voy a updatear:")
        im = self.cam.getImage()
        print("\t - Ya tengo una imagen para estimar")
        coords, im = self.estimate(im)
        print("\t - Ya tengo la estimacion")

        self.gui.display_result(im, coords)
        print("\t - Deberia estar en pantalla")

