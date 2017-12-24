#!/usr/bin/env python

"""
pose_estimator.py: Module for estimating human pose with Caffe.
Based on @shihenw code:
https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/python/demo.ipynb
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/01"

import math

import caffe
import cv2 as cv
import numpy as np


class PoseEstimator:
    def __init__(self, model, weights):
        """
        Constructs Estimator class.
        @param model: Caffe model
        @param weights: Caffe model weights
        """
        self.net = caffe.Net(model, weights, caffe.TEST)

    @staticmethod
    def get_boxes(im, coords, size):
        """
        Crops the original image once for each person.
        @param im: np.array - input image
        @param coords: np.array - human coordinates
        @param size: int - size of the squared box
        @return: np.array - cropped images around each person
        """
        num_people = coords.shape[0]
        boxes = np.ones((num_people, size, size, 3)) * 128

        pad_h = np.ones((im.shape[0] + size, size / 2, 3)) * 128
        pad_v = np.ones((size / 2, im.shape[1], 3)) * 128
        im = np.vstack((pad_v, im, pad_v))
        im = np.hstack((pad_h, im, pad_h))
        for i in range(num_people):
            y = coords[i][1] + size / 2
            x = coords[i][0] + size / 2
            boxes[i, :, :, :] = im[y - size / 2: y + size / 2,
                                x - size / 2: x + size / 2]
        return boxes

    @staticmethod
    def gen_gaussmap(size, sigma):
        """
        Generates a grayscale image with a centered Gaussian
        @param size: int - map size
        @param sigma: float - Gaussian sigma
        @return: np.array - Gaussian map
        """
        gaussmap = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                dist_sq = (x - size / 2) * (x - size / 2) \
                          + (y - size / 2) * (y - size / 2)
                exponent = dist_sq / 2.0 / sigma / sigma
                gaussmap[y, x] = math.exp(-exponent)

        return gaussmap

    def estimate(self, im, gaussmap):
        """
        Estimates human pose.
        @param im: np.array - input image
        @param gaussmap: np.array - Gaussian map
        @return: np.array: articulations coordinates
        """
        # Adds gaussian map channel to the input
        input_4ch = np.ones((im.shape[0], im.shape[1], 4))
        input_4ch[:, :, 0:3] = im / 256.0 - 0.5  # normalize to [-0.5, 0.5]
        input_4ch[:, :, 3] = gaussmap

        # Adapts input to the net
        input_adapted = np.transpose(np.float32(input_4ch[:, :, :, np.newaxis]),
                                     (3, 2, 0, 1))
        self.net.blobs['data'].reshape(*input_adapted.shape)
        self.net.blobs['data'].data[...] = input_adapted

        # Estimates the pose
        output_blobs = self.net.forward()
        pose_map = np.squeeze(self.net.blobs[output_blobs.keys()[0]].data)

        return pose_map

    @staticmethod
    def get_coords(joint_map, person_coord, r, boxsize):
        """
        Get joint coordinates and resize them given a heatmap.
        @param joint_map: np.array - heatmap
        @param person_coord: np.array - person center coordinates
        @param r: float - resize rate
        @param boxsize: int - boxsize
        @return: joint coordinates
        """
        # Get coordinates
        joint_coord = list(
            np.unravel_index(joint_map.argmax(), joint_map.shape))
        # Back to full coordinates
        joint_coord[0] = (joint_coord[0] - (boxsize / 2) + person_coord[1]) / r
        joint_coord[1] = (joint_coord[1] - (boxsize / 2) + person_coord[0]) / r

        return joint_coord

    @staticmethod
    def draw_limbs(limbs, im, pose_coords):
        """
        Draw limbs over the original image.
        @param limbs: list - relationship between limbs and joints
        @param im: np.array - original image
        @param pose_coords: np.array - coordinates of the predicted
        joints
        @return: drawn image
        """
        stickwidth = 6
        colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0],
                  [170, 255, 0], [255, 170, 0], [255, 0, 0], [255, 0, 170],
                  [170, 0, 255]]  # note BGR ...

        for joint_coords in pose_coords:
            for joint_coord in joint_coords[:-1]:
                cv.circle(im, (int(joint_coord[1]), int(joint_coord[0])), 3,
                          (0, 0, 0), -1)

            for l in range(limbs.shape[0]):
                x = [joint_coords[limbs[l][0] - 1][0],
                     joint_coords[limbs[l][1] - 1][0]]
                y = [joint_coords[limbs[l][0] - 1][1],
                     joint_coords[limbs[l][1] - 1][1]]

                m_x = np.mean(x)
                m_y = np.mean(y)

                length = ((x[0] - x[1]) ** 2. + (y[0] - y[1]) ** 2.) ** 0.5
                angle = math.degrees(math.atan2(x[0] - x[1], y[0] - y[1]))
                polygon = cv.ellipse2Poly((int(m_y), int(m_x)),
                                          (int(length / 2), stickwidth),
                                          int(angle), 0, 360, 1)
                cv.fillConvexPoly(im, polygon, colors[l])

        return im
