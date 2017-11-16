#!/usr/bin/env python

"""
gui.py: GUI class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import cv2
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import pyqtSignal


class GUI(QWidget):

    updGUI = pyqtSignal()

    def __init__(self, parent=None):
        """
        GUI class creates the GUI that we're going to use to preview
        the live video as well as the estimation results.
        :param parent: bool
        """
        QWidget.__init__(self, parent)

        self.setWindowTitle("Human pose estimation !")
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


    def setCamera(self, cam):
        """
        Set camera.
        :param cam: Camera object
        """
        self.cam = cam


    def setEstimator(self, estimator):
        """
        Set estimator.
        :param estimator: Estimator object
        """
        self.estimator = estimator


    def update(self):
        """
        Updates the GUI.
        """
        # Get original image and display it
        im_live_pre = self.cam.getImage()
        im_live = QImage(im_live_pre, im_live_pre.shape[1],
                         im_live_pre.shape[0], QImage.Format_RGB888)
        im_live = QPixmap.fromImage(im_live)
        self.im_live_label.setPixmap(im_live)


    def display_result(self, im, pose_coords):
        """
        Display image with estimated limbs and joint coordinates
        :param im: np.array - Image with limbs
        :param pose_coords: np.array - Pose coordinates
        :return:
        """
        im = QImage(im, im.shape[1], im.shape[0], QImage.Format_RGB888)
        im = QPixmap.fromImage(im)
        self.im_pose_label.setPixmap(im)