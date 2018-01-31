#!/usr/bin/env python

"""
gui.py: GUI class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QLabel


class GUI(QWidget):
    updGUI = pyqtSignal()

    def __init__(self, cam, parent=None):
        """
        GUI class creates the GUI that we're going to use to preview
        the live video as well as the estimation results.
        @param parent: bool
        """
        QWidget.__init__(self, parent)

        self.setWindowTitle("Human pose estimation")
        self.resize(1400, 800)
        self.move(10, 20)
        self.updGUI.connect(self.update)

        # Original image label.
        self.im_live_label = QLabel(self)
        self.im_live_label.resize(640, 480)
        self.im_live_label.move(30, 30)
        self.im_live_label.show()

        # Pose image label.
        self.im_pose_label = QLabel(self)
        self.im_pose_label.resize(640, 480)
        self.im_pose_label.move(700, 30)
        self.im_pose_label.show()
        self.im_pose = np.zeros((480, 640), dtype=np.uint8)

        # Camera
        self.cam = cam

    # noinspection PyArgumentList
    def update(self):
        """ Updates the GUI. """
        # Get original image and display it
        im_live = self.cam.get_image()
        im_live = QImage(im_live, im_live.shape[1], im_live.shape[0],
                         QImage.Format_RGB888)
        # noinspection PyCallByClass
        im_live = QPixmap.fromImage(im_live)
        self.im_live_label.setPixmap(im_live)

        im_pose = QImage(self.im_pose, self.im_pose.shape[1],
                         self.im_pose.shape[0], QImage.Format_RGB888)
        im_pose = QPixmap.fromImage(im_pose)
        self.im_pose_label.setPixmap(im_pose)
