#!/usr/bin/env python

"""
humanpose.py: Receive images from live video and estimate human pose.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import os

# Avoids verbosity when loading Caffe model
os.environ['GLOG_minloglevel'] = '2'

import signal
import sys
from PyQt5 import QtWidgets

from Camera.camera import Camera
from Camera.threadcamera import ThreadCamera
from Estimator.estimator import Estimator
from Estimator.threadestimator import ThreadEstimator
from GUI.gui import GUI
from GUI.threadgui import ThreadGUI

signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == '__main__':
    # Init objects
    app = QtWidgets.QApplication(sys.argv)

    cam = Camera()
    window = GUI(cam)
    estimator = Estimator(window, cam)
    window.show()

    # Threading camera
    t_cam = ThreadCamera(cam)
    t_cam.start()

    # Threading estimator
    t_estimator = ThreadEstimator(estimator)
    t_estimator.start()

    # Threading GUI
    t_gui = ThreadGUI(window)
    t_gui.start()

    sys.exit(app.exec_())
