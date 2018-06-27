#!/usr/bin/env python

"""
human.py: Human detection base class.
"""

import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter

__author__ = "David Pascual Hernandez"
__date__ = "2018/06/04"


def get_sample_ready(im, boxsize):
    # The image is resized to conveniently crop a region that can
    # nicely fit a person
    scale = boxsize / float(im.shape[0])
    im_nopad = cv2.resize(im, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # Padded to become multiple of 8 (downsize factor of the CPM)
    im_padded, pad = padRightDownCorner(im_nopad)

    return im_padded, scale


def padRightDownCorner(img):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % 8 == 0) else 8 - (h % 8)  # down
    pad[3] = 0 if (w % 8 == 0) else 8 - (w % 8)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + 128, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * + 128, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * + 128, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + 128, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


class Human:
    def __init__(self, boxsize=192):
        self.im = None
        self.heatmap = None
        self.boxsize = boxsize

    def init_net(self):
        pass

    def detect(self):
        pass

    def peaks_coords(self):
        """
        Gets the exact coordinates of each person in the heatmap.
        @return: np.array - people coordinates
        """
        # Founds the peaks in the output
        # noinspection PyUnresolvedReferences
        data_max = maximum_filter(self.heatmap, 3)
        peaks = (self.heatmap == data_max)
        thresh = (data_max > 0.5)
        peaks[thresh == 0] = 0

        # Peaks coordinates
        x = np.nonzero(peaks)[1]
        y = np.nonzero(peaks)[0]
        peaks_coords = []
        for x_coord, y_coord in zip(x, y):
            peaks_coords.append([x_coord, y_coord])

        return np.array(peaks_coords)

    def get_bboxes(self, im):
        """
        Get human bounding boxes coordinates.
        :param im: np.array - input image
        :param boxsize: int - image size used for inference
        :return: np.array - bounding boxes coordinates.
        """
        self.im, scale = get_sample_ready(im, self.boxsize)

        self.detect()

        person_coords = self.peaks_coords()

        bboxes = []
        for cx, cy in person_coords:
            upper_left = (cx - self.boxsize / 2, cy - self.boxsize / 2)
            bottom_right = (cx + self.boxsize / 2, cy + self.boxsize / 2)

            bboxes.append((upper_left, bottom_right))

        bboxes = np.array(bboxes) / scale

        return bboxes.astype(np.int64)
