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
    def __init__(self, gui, cam, data):
        """
        Estimator class gets human pose estimations for a given image.
        @param gui: GUI object
        @param cam: Camera object
        @param data: parsed YAML config. file
        """
        self.data = data
        self.config = data["Settings"]
        print(self.config["boxsize"])
        self.framework = self.data["Framework"]

        if self.framework == "caffe":
            self.deploy_models = caffe_cpm.load_model(
                self.config["caffe_models"])

        elif self.framework == "tensorflow":
            self.model_paths = tf_cpm.load_model(self.config["tf_models"])
            self.tf_config = tf_cpm.set_dev(self.config)

        else:
            print(self.framework, " framework is not supported")
            print("Available frameworks: 'caffe', 'tensorflow'")
            exit()

        self.gui = gui
        self.cam = cam
        self.set_caffe = False

    def estimate(self, im):
        """
        Estimate human pose.
        @param im: np.array - Image, preferably with humans
        @return: np.array, np.array - joint coordinates & limbs drawn
        over original image
        """
        im_predicted, pose_coords = [None, None]
        if self.framework == "caffe":
            if not self.set_caffe:
                caffe_cpm.set_dev(self.config)
                self.set_caffe = True

            im_predicted, pose_coords = caffe_cpm.predict(im, self.config,
                                                          self.deploy_models,
                                                          viz=False)

        elif self.framework == "tensorflow":
            im_predicted, pose_coords = tf_cpm.predict(im, self.tf_config,
                                                       self.model_paths,
                                                       self.config, viz=False)

        else:
            print(self.framework, " framework is not supported")
            print("Available frameworks: 'caffe', 'tensorflow'")
            exit()

        return im_predicted, pose_coords

    def update(self):
        """ Update estimator. """
        im = self.cam.get_image()
        im, coords = self.estimate(im)

        self.gui.im_pose = im
