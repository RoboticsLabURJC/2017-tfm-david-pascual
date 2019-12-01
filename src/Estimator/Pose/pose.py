#!/usr/bin/env python

"""
pose.py: Pose estimation base class
"""
__author__ = "David Pascual Hernandez"
__date__ = "2019/11/28"


class PoseEstimator:
    def __init__(self, model_fname, boxsize, confidence_th=0.3):
        """ Constructs Pose Estimator class. """
        self.model_fname = model_fname
        self.net = None

        self.im = None
        self.boxsize = boxsize

        self.confidence_th = confidence_th

    def init_net(self):
        pass

    def estimate(self):
        """ Estimates human pose. """
        pass

    def get_coords(self, sample, human_bbox):
        """ Estimate human pose given an input image. """
        pass
