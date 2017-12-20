#!/usr/bin/env python

"""
estimator.py: Estimator class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import cv2

from Caffe import caffe_cpm as cpm


class Estimator:
    def __init__(self, gui, cam):
        """
        Estimator class gets human pose estimations for a given image.
        """
        print("\nLoading Caffe models...")
        self.model, self.deploy_models = cpm.load_model()
        print("loaded\n")

        self.gui = gui
        self.cam = cam

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

    def update(self):
        """ Update estimator. """
        im = self.cam.get_image()
        coords, im = self.estimate(im)

        self.gui.display_result(im)
