#!/usr/bin/env python

"""
estimator.py: Estimator class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import sys

import easyiceconfig as easy_ice

from Caffe import cpm


class Estimator:
    def __init__(self, gui, cam):
        """
        Estimator class gets human pose estimations for a given image.
        """
        ic = easy_ice.initialize(sys.argv)
        self.framework = ic.getProperties().getProperty("Humanpose.Framework")

        if self.framework == "caffe":
            self.param_conf, self.model_conf = cpm.read_settings()

            cpm.set_dev(self.param_conf)
            self.deploy_models = cpm.load_model(self.model_conf)

        elif self.framework == "tensorflow":
            print(self.framework)

        else:
            print(self.framework, " framework is not supported")
            print("Available frameworks: 'caffe', 'tensorflow'")
            exit()

        self.gui = gui
        self.cam = cam

    def estimate(self, im):
        """
        Estimate human pose.
        @param im: np.array - Image, preferably with humans
        @return: np.array, np.array - joint coordinates & limbs drawn
        over original image
        """
        pose_coords, im_predicted = cpm.predict(self.model_conf,
                                                self.deploy_models,
                                                im, viz=False)

        return pose_coords, im_predicted

    def update(self):
        """ Update estimator. """
        im = self.cam.get_image()
        coords, im = self.estimate(im)

        self.gui.display_result(im)
