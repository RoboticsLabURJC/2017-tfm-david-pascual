#!/usr/bin/env python

"""
gui.py: GUI class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import cv2
import numpy as np

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton


class GUI3D(QWidget):
    updGUI = pyqtSignal()

    def __init__(self, cam, cam_depth, parent=None):
        """
        GUI class creates the GUI that we"re going to use to preview
        the live video as well as the estimation results.
        @param cam: Camera object
        @param estimator: Estimator object
        @param parent: bool
        """
        self.cam = cam
        self.cam_depth = cam_depth

        w, h = (640, 480)
        self.im_depth = np.zeros((h, w, 3), dtype=np.uint8)

        self.live = False
        self.display = False

        QWidget.__init__(self, parent)
        self.setWindowTitle("JdeRobot - Human Pose Estimation w/ CPMs")
        self.resize(1550, 610)
        self.move(150, 50)
        self.setWindowIcon(QIcon("GUI/resources/jderobot.png"))
        self.updGUI.connect(self.update)

        # Original image label.
        self.im_label = QLabel(self)
        self.im_label.resize(w, h)
        self.im_label.move(25, 90)
        self.im_label.show()

        # Depth image label.
        self.im_depth_label = QLabel(self)
        self.im_depth_label.resize(w, h)
        self.im_depth_label.move(885, 90)
        self.im_depth_label.show()

        # Logo
        self.logo_label = QLabel(self)
        self.logo_label.resize(150, 150)
        self.logo_label.move(700, 400)
        self.logo_label.setScaledContents(True)

        logo_img = QImage()
        logo_img.load("GUI/resources/jderobot.png")
        self.logo_label.setPixmap(QPixmap.fromImage(logo_img))
        self.logo_label.show()

    # noinspection PyArgumentList
    def update(self):
        """ Updates the GUI. """
        # Get original image and display it
        im = self.cam.get_image()
        h, w, d = im.shape
        im = QImage(im, w, h, QImage.Format_RGB888)
        im = im.scaled(self.im_label.size())
        # noinspection PyCallByClass
        im = QPixmap.fromImage(im)
        self.im_label.setPixmap(im)

        im_depth = self.cam_depth.get_image()
        cv2.normalize(im_depth, im_depth, 0, 1, cv2.NORM_MINMAX)
        im_depth *= 255
        im_depth = np.dstack((im_depth, im_depth, im_depth)).astype(np.uint8)

        h, w, d = im_depth.shape
        im_depth = QImage(im_depth, w, h, QImage.Format_RGB888)
        im_depth = im_depth.scaled(self.im_depth_label.size())
        im_depth = QPixmap.fromImage(im_depth)
        self.im_depth_label.setPixmap(im_depth)