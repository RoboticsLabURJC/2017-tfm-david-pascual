#!/usr/bin/env python

"""
humanpose.py: Receive images from live video and estimate human pose.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import sys
import signal

from PyQt5 import QtWidgets

from Camera.camera import Camera
from Camera.threadcamera import ThreadCamera
from GUI.gui import GUI
from GUI.threadgui import ThreadGUI
from Estimator.estimator import Estimator
from Estimator.threadestimator import ThreadEstimator

signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == '__main__':
    cam = Camera()
    app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    window.setCamera(cam)
    window.show()
    estimator = Estimator()
    estimator.setGUI(window)
    estimator.setCamera(cam)
    
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

