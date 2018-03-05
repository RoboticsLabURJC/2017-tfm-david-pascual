#!/usr/bin/env python

"""
threadcamera.py: ThreadCamera class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import threading
import time
from datetime import datetime

t_cycle = 100  # ms


class ThreadCamera(threading.Thread):
    def __init__(self, cam):
        """
        Threading class for Camera.
        @param cam: Camera object
        """
        self.cam = cam
        threading.Thread.__init__(self)

    def run(self):
        """ Updates the thread. """
        while True:
            start_time = datetime.now()
            self.cam.update()
            end_time = datetime.now()

            dt = end_time - start_time
            dtms = ((dt.days * 24 * 60 * 60 + dt.seconds) * 1000
                    + dt.microseconds / 1000.0)

            if dtms < t_cycle:
                time.sleep((t_cycle - dtms) / 1000.0)
