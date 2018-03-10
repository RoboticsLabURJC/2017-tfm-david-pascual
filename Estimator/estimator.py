#!/usr/bin/env python

"""
estimator.py: Estimator class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

from Caffe import cpm as caffe_cpm
from TensorFlow import cpm as tf_cpm


class Estimator:
    def __init__(self, cam, gui, data):
        """
        Estimator class gets human pose estimations for a given image.
        @param cam: Camera object
        @param gui: GUI object
        @param data: parsed YAML config. file
        """
        self.cam = cam
        self.gui = gui
        self.live = gui.live

        self.data = data
        self.config = data["Settings"]
        self.framework = self.data["Framework"]

        if self.framework == "caffe":
            self.models = caffe_cpm.load_model(self.config["caffe_models"])

        elif self.framework == "tensorflow":
            im_shape = (self.cam.im_height, self.cam.im_width, 3)
            pose_shape = (self.config["boxsize"], self.config["boxsize"], 15)

            self.tf_config = tf_cpm.set_dev(self.config["GPU"])
            self.models = tf_cpm.load_model(self.config, im_shape,
                                            self.tf_config)

        else:
            print(self.framework, " framework is not supported")
            print("Available frameworks: 'caffe', 'tensorflow'")
            exit()

        self.caffe_set = False


    def estimate(self, im):
        """
        Estimate human pose.
        @param im: np.array - Image, preferably with humans
        @return: np.array, np.array - joint coordinates & limbs drawn
        over original image
        """
        im_predicted, pose_coords = [None, None]
        if self.framework == "caffe":
            if not self.caffe_set:
                caffe_cpm.set_dev(self.config["GPU"])
                self.caffe_set = True

            im_predicted, pose_coords, _ = caffe_cpm.predict(im, self.config,
                                                             self.models,
                                                             viz=False)

        if self.framework == "tensorflow":
            im_predicted, pose_coords, _ = tf_cpm.predict(im, self.models,
                                                          self.config,
                                                          viz=False)

        return im_predicted, pose_coords

    def update(self):
        """ Update estimator. """
        im = self.cam.get_image()
        im, coords = self.estimate(im)

        self.gui.im_pred = im
