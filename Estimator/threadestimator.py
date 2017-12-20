#!/usr/bin/env python

"""
threadestimator.py: Thread for Estimator class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import threading


class ThreadEstimator(threading.Thread):
    def __init__(self, estimator):
        """
        Threading class for estimator.
        :param estimtor: Estimator object
        """
        self.estimator = estimator
        threading.Thread.__init__(self)

    def run(self):
        """ Updates the thread. """
        while (True):
            self.estimator.update()
