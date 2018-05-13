#!/usr/bin/env python

"""
estimator.py: Estimator class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import numpy as np

from Frameworks.Caffe import cpm as caffe_cpm
from Frameworks.TensorFlow import cpm as tf_cpm


class Estimator:
    def __init__(self, cam, viz3d, gui, data):
        """
        Estimator class gets human pose estimations for a given image.
        @param cam: Camera object
        @param viz3d: Viz3D object
        @param gui: GUI object
        @param data: parsed YAML config. file
        """
        self.cam = cam
        self.viz3d = viz3d
        self.gui = gui

        self.data = data
        self.config = data["Settings"]
        self.framework = self.data["Framework"]

        if self.framework == "caffe":
            self.models = caffe_cpm.load_model(self.config["caffe_models"])

        elif self.framework == "tensorflow":
            im_shape = (self.cam.im_height, self.cam.im_width, 3)

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

    def get_depth_point(self, point, depth):
        """
        Get joints depth information.
        """
        y, x = point

        x = np.clip(int(x), 0, depth.shape[0])
        y = depth.shape[0] - np.clip(int(y), 0, depth.shape[0])

        # wsize = 11
        # z = np.median(depth[wsize / 2 - y:wsize / 2 + y,
        #                     wsize / 2 - x:wsize / 2 + x])

        z = depth[y, x]

        return np.array((x, z, y))

    def update(self):
        """ Update estimator. """
        im, im_depth = self.cam.get_image()
        im, coords = self.estimate(im)

        self.gui.im_pred = im

        if len(coords) and self.gui.display:
            for human_coords in coords:
                limbs = np.array(self.config["limbs"]).reshape((-1, 2)) - 1
                for l, (p, q) in enumerate(limbs):
                    point_a = self.get_depth_point(human_coords[p], im_depth)
                    point_b = self.get_depth_point(human_coords[q], im_depth)

                    color = self.config["colors"][l]
                    self.viz3d.drawSegment(point_a, point_b, color)

                for joint in human_coords[:-1]:
                    point = self.get_depth_point(joint, im_depth)
                    self.viz3d.drawPoint(point, (255, 255, 255))

            self.gui.display = False
