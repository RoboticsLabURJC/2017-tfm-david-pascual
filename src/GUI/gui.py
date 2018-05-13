#!/usr/bin/env python

"""
gui.py: GUI class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton


class GUI(QWidget):
    updGUI = pyqtSignal()

    def __init__(self, cam, parent=None):
        """
        GUI class creates the GUI that we"re going to use to preview
        the live video as well as the estimation results.
        @param cam: Camera object
        @param estimator: Estimator object
        @param parent: bool
        """
        self.cam = cam
        self.im_pred, _ = self.cam.get_image()
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
        self.im_label.resize(640, 480)
        self.im_label.move(25, 90)
        self.im_label.show()

        # Processed image label.
        self.im_pred_label = QLabel(self)
        self.im_pred_label.resize(640, 480)
        self.im_pred_label.move(885, 90)
        self.im_pred_label.show()

        # Button for configuring detection flow
        self.button_cont_detection = QPushButton(self)
        self.button_cont_detection.move(725, 100)
        self.button_cont_detection.clicked.connect(self.pred_toggle)
        self.button_cont_detection.setStyleSheet("QPushButton"
                                                 "{color: green;}")
        self.button_cont_detection.setText("Switch on\nContinuous"
                                           "\nDetection")

        # Button for processing a single frame
        self.button_one_frame = QPushButton(self)
        self.button_one_frame.move(725, 200)
        self.button_one_frame.clicked.connect(self.single_pred)
        self.button_one_frame.setText("On-demand\ndetection")

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
        im, _ = self.cam.get_image()
        im = QImage(im, im.shape[1], im.shape[0], QImage.Format_RGB888)
        # noinspection PyCallByClass
        im = QPixmap.fromImage(im)
        self.im_label.setPixmap(im)

        if self.live:
            im_pred = QImage(self.im_pred, self.im_pred.shape[1],
                             self.im_pred.shape[0], QImage.Format_RGB888)
            im_pred = QPixmap.fromImage(im_pred)
            self.im_pred_label.setPixmap(im_pred)

            self.display = True

    def pred_toggle(self):
        """
        Toggles between live and single prediction modes.
        """
        self.live = not self.live

        if self.live:
            self.button_cont_detection.setStyleSheet("QPushButton"
                                                     "{color: red;}")
            self.button_cont_detection.setText("Switch off\nContinuous"
                                               "\nDetection")
        else:
            self.button_cont_detection.setStyleSheet("QPushButton"
                                                     "{color: green;}")
            self.button_cont_detection.setText("Switch on\nContinuous"
                                               "\nDetection")

    def single_pred(self):
        """
        Single prediction.
        """
        im_pred = QImage(self.im_pred, self.im_pred.shape[1],
                         self.im_pred.shape[0], QImage.Format_RGB888)
        im_pred = QPixmap.fromImage(im_pred)
        self.im_pred_label.setPixmap(im_pred)

        self.display = True
