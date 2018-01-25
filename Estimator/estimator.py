#!/usr/bin/env python

"""
estimator.py: Estimator class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import sys

import easyiceconfig as easy_ice

# from Caffe import cpm as caffe_cpm
from TensorFlow import cpm as tf_cpm


class Estimator:
    def __init__(self, gui, cam):
        """
        Estimator class gets human pose estimations for a given image.
        """
        ic = easy_ice.initialize(sys.argv)
        self.framework = ic.getProperties().getProperty("Humanpose.Framework")

        if self.framework == "caffe":
            # self.param_conf, self.model_conf = caffe_cpm.read_settings()

            # caffe_cpm.set_dev(self.param_conf)
            # self.deploy_models = caffe_cpm.load_model(self.model_conf)
            print("Cerrado por vacaciones")

        elif self.framework == "tensorflow":
            self.boxsize = 184
            self.model_paths = tf_cpm.load_model("Estimator/TensorFlow/model/")
            self.tf_config = tf_cpm.set_dev()

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
        if self.framework == "caffe":
            # pose_coords, im_predicted = caffe_cpm.predict(self.model_conf,
            #                                               self.deploy_models,
            #                                               im, viz=False)
            print("Cerrado por vacaciones")

        elif self.framework == "tensorflow":
            im_predicted, pose_coords = tf_cpm.predict(im, self.tf_config,
                                                       self.model_paths,
                                                       self.boxsize, viz=False)

        else:
            print(self.framework, " framework is not supported")
            print("Available frameworks: 'caffe', 'tensorflow'")
            exit()

        return pose_coords, im_predicted

    def update(self):
        """ Update estimator. """
        im = self.cam.get_image()
        coords, im = self.estimate(im)

        self.gui.display_result(im)
