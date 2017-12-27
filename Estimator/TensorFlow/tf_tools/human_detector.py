#!/usr/bin/env python

"""
human_detector.py: Module for human detection with TensorFlow.
Based on @psycharm code:
https://github.com/psycharo/cpm/blob/master/example.ipynb
"""

__author__ = "David Pascual Hernandez"
__date__ = "2017/mm/dd"

import math
import time

import cv2
import numpy as np
import scipy
import tensorflow as tf
import tensorflow.contrib.layers as layers


def inference_person(image):
    """
    Human detection model.
    @param image: np.array - human image
    @return: TensorFlow model
    """
    with tf.variable_scope('PersonNet'):
        conv1_1 = layers.conv2d(image, 64, 3, 1, activation_fn=None,
                                scope='conv1_1')
        conv1_1 = tf.nn.relu(conv1_1)
        conv1_2 = layers.conv2d(conv1_1, 64, 3, 1, activation_fn=None,
                                scope='conv1_2')
        conv1_2 = tf.nn.relu(conv1_2)
        pool1_stage1 = layers.max_pool2d(conv1_2, 2, 2)
        conv2_1 = layers.conv2d(pool1_stage1, 128, 3, 1, activation_fn=None,
                                scope='conv2_1')
        conv2_1 = tf.nn.relu(conv2_1)
        conv2_2 = layers.conv2d(conv2_1, 128, 3, 1, activation_fn=None,
                                scope='conv2_2')
        conv2_2 = tf.nn.relu(conv2_2)
        pool2_stage1 = layers.max_pool2d(conv2_2, 2, 2)
        conv3_1 = layers.conv2d(pool2_stage1, 256, 3, 1, activation_fn=None,
                                scope='conv3_1')
        conv3_1 = tf.nn.relu(conv3_1)
        conv3_2 = layers.conv2d(conv3_1, 256, 3, 1, activation_fn=None,
                                scope='conv3_2')
        conv3_2 = tf.nn.relu(conv3_2)
        conv3_3 = layers.conv2d(conv3_2, 256, 3, 1, activation_fn=None,
                                scope='conv3_3')
        conv3_3 = tf.nn.relu(conv3_3)
        conv3_4 = layers.conv2d(conv3_3, 256, 3, 1, activation_fn=None,
                                scope='conv3_4')
        conv3_4 = tf.nn.relu(conv3_4)
        pool3_stage1 = layers.max_pool2d(conv3_4, 2, 2)
        conv4_1 = layers.conv2d(pool3_stage1, 512, 3, 1, activation_fn=None,
                                scope='conv4_1')
        conv4_1 = tf.nn.relu(conv4_1)
        conv4_2 = layers.conv2d(conv4_1, 512, 3, 1, activation_fn=None,
                                scope='conv4_2')
        conv4_2 = tf.nn.relu(conv4_2)
        conv4_3 = layers.conv2d(conv4_2, 512, 3, 1, activation_fn=None,
                                scope='conv4_3')
        conv4_3 = tf.nn.relu(conv4_3)
        conv4_4 = layers.conv2d(conv4_3, 512, 3, 1, activation_fn=None,
                                scope='conv4_4')
        conv4_4 = tf.nn.relu(conv4_4)
        conv5_1 = layers.conv2d(conv4_4, 512, 3, 1, activation_fn=None,
                                scope='conv5_1')
        conv5_1 = tf.nn.relu(conv5_1)
        conv5_2_CPM = layers.conv2d(conv5_1, 128, 3, 1, activation_fn=None,
                                    scope='conv5_2_CPM')
        conv5_2_CPM = tf.nn.relu(conv5_2_CPM)
        conv6_1_CPM = layers.conv2d(conv5_2_CPM, 512, 1, 1, activation_fn=None,
                                    scope='conv6_1_CPM')
        conv6_1_CPM = tf.nn.relu(conv6_1_CPM)
        conv6_2_CPM = layers.conv2d(conv6_1_CPM, 1, 1, 1, activation_fn=None,
                                    scope='conv6_2_CPM')
        concat_stage2 = tf.concat(axis=3, values=[conv6_2_CPM, conv5_2_CPM])
        Mconv1_stage2 = layers.conv2d(concat_stage2, 128, 7, 1,
                                      activation_fn=None, scope='Mconv1_stage2')
        Mconv1_stage2 = tf.nn.relu(Mconv1_stage2)
        Mconv2_stage2 = layers.conv2d(Mconv1_stage2, 128, 7, 1,
                                      activation_fn=None, scope='Mconv2_stage2')
        Mconv2_stage2 = tf.nn.relu(Mconv2_stage2)
        Mconv3_stage2 = layers.conv2d(Mconv2_stage2, 128, 7, 1,
                                      activation_fn=None, scope='Mconv3_stage2')
        Mconv3_stage2 = tf.nn.relu(Mconv3_stage2)
        Mconv4_stage2 = layers.conv2d(Mconv3_stage2, 128, 7, 1,
                                      activation_fn=None, scope='Mconv4_stage2')
        Mconv4_stage2 = tf.nn.relu(Mconv4_stage2)
        Mconv5_stage2 = layers.conv2d(Mconv4_stage2, 128, 7, 1,
                                      activation_fn=None, scope='Mconv5_stage2')
        Mconv5_stage2 = tf.nn.relu(Mconv5_stage2)
        Mconv6_stage2 = layers.conv2d(Mconv5_stage2, 128, 1, 1,
                                      activation_fn=None, scope='Mconv6_stage2')
        Mconv6_stage2 = tf.nn.relu(Mconv6_stage2)
        Mconv7_stage2 = layers.conv2d(Mconv6_stage2, 1, 1, 1,
                                      activation_fn=None, scope='Mconv7_stage2')
        concat_stage3 = tf.concat(axis=3, values=[Mconv7_stage2, conv5_2_CPM])
        Mconv1_stage3 = layers.conv2d(concat_stage3, 128, 7, 1,
                                      activation_fn=None, scope='Mconv1_stage3')
        Mconv1_stage3 = tf.nn.relu(Mconv1_stage3)
        Mconv2_stage3 = layers.conv2d(Mconv1_stage3, 128, 7, 1,
                                      activation_fn=None, scope='Mconv2_stage3')
        Mconv2_stage3 = tf.nn.relu(Mconv2_stage3)
        Mconv3_stage3 = layers.conv2d(Mconv2_stage3, 128, 7, 1,
                                      activation_fn=None, scope='Mconv3_stage3')
        Mconv3_stage3 = tf.nn.relu(Mconv3_stage3)
        Mconv4_stage3 = layers.conv2d(Mconv3_stage3, 128, 7, 1,
                                      activation_fn=None, scope='Mconv4_stage3')
        Mconv4_stage3 = tf.nn.relu(Mconv4_stage3)
        Mconv5_stage3 = layers.conv2d(Mconv4_stage3, 128, 7, 1,
                                      activation_fn=None, scope='Mconv5_stage3')
        Mconv5_stage3 = tf.nn.relu(Mconv5_stage3)
        Mconv6_stage3 = layers.conv2d(Mconv5_stage3, 128, 1, 1,
                                      activation_fn=None, scope='Mconv6_stage3')
        Mconv6_stage3 = tf.nn.relu(Mconv6_stage3)
        Mconv7_stage3 = layers.conv2d(Mconv6_stage3, 1, 1, 1,
                                      activation_fn=None, scope='Mconv7_stage3')
        concat_stage4 = tf.concat(axis=3, values=[Mconv7_stage3, conv5_2_CPM])
        Mconv1_stage4 = layers.conv2d(concat_stage4, 128, 7, 1,
                                      activation_fn=None, scope='Mconv1_stage4')
        Mconv1_stage4 = tf.nn.relu(Mconv1_stage4)
        Mconv2_stage4 = layers.conv2d(Mconv1_stage4, 128, 7, 1,
                                      activation_fn=None, scope='Mconv2_stage4')
        Mconv2_stage4 = tf.nn.relu(Mconv2_stage4)
        Mconv3_stage4 = layers.conv2d(Mconv2_stage4, 128, 7, 1,
                                      activation_fn=None, scope='Mconv3_stage4')
        Mconv3_stage4 = tf.nn.relu(Mconv3_stage4)
        Mconv4_stage4 = layers.conv2d(Mconv3_stage4, 128, 7, 1,
                                      activation_fn=None, scope='Mconv4_stage4')
        Mconv4_stage4 = tf.nn.relu(Mconv4_stage4)
        Mconv5_stage4 = layers.conv2d(Mconv4_stage4, 128, 7, 1,
                                      activation_fn=None, scope='Mconv5_stage4')
        Mconv5_stage4 = tf.nn.relu(Mconv5_stage4)
        Mconv6_stage4 = layers.conv2d(Mconv5_stage4, 128, 1, 1,
                                      activation_fn=None, scope='Mconv6_stage4')
        Mconv6_stage4 = tf.nn.relu(Mconv6_stage4)
        Mconv7_stage4 = layers.conv2d(Mconv6_stage4, 1, 1, 1,
                                      activation_fn=None, scope='Mconv7_stage4')
    return Mconv7_stage4


class HumanDetector:
    def __init__(self, im, config, model_path, boxsize):
        """
        Class for human detection.
        @param im: image to be analyzed
        @param config: TensorFlow configuration
        @param model_path: human detection model path
        @param boxsize: int - boxsize
        """
        self.im_original = im

        factor = boxsize / float(im.shape[0])
        self.im = cv2.resize(im, None, fx=factor, fy=factor,
                             interpolation=cv2.INTER_CUBIC)
        self.im = self.im[np.newaxis] / 255.0 - 0.5

        self.config = config
        self.model_path = model_path
        self.boxsize = boxsize

    @staticmethod
    def set_model():
        """
        Get the model ready for inference.
        @return: Model restorer
        """
        model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  'CPM/PersonNet')
        restorer = tf.train.Saver(model)

        return restorer

    def get_map(self):
        """
        Get human heatmap.
        @return: np.array - human heatmap
        """
        with tf.variable_scope('CPM'):
            # Input dims for the human model
            image_in = tf.placeholder(tf.float32, self.im.shape)

            map_human = inference_person(image_in)
            map_human_large = tf.image.resize_images(map_human,
                                                     [self.im.shape[1],
                                                      self.im.shape[2]])

        restorer = self.set_model()

        with tf.Session(config=self.config) as sess:
            restorer.restore(sess, self.model_path)
            start_time = time.time()
            map_human = sess.run(map_human_large, {image_in: self.im})
            print("Human detected! (%.2f ms)" % (1000 * (time.time()
                                                         - start_time)))

        return map_human

    def get_humans_coords(self, map_human_full):
        """
        Get human coordinates given a human heatmap.
        @param map_human_full: np.array - full human heatmap
        @return: np.array - human coordinates
        """
        # noinspection PyUnresolvedReferences

        map_human_full = np.squeeze(map_human_full, 0)
        data_max = scipy.ndimage.filters.maximum_filter(map_human_full, 3)
        peaks = (map_human_full == data_max)

        thresh = (data_max > 0.5)
        peaks[thresh == 0] = 0

        # Peaks coordinates
        x = np.nonzero(peaks)[1]
        y = np.nonzero(peaks)[0]

        humans_coords = []
        for x_coord, y_coord in zip(x, y):
            humans_coords.append([x_coord, y_coord])
        return np.array(humans_coords)

    def gen_gaussmap(self, sigma):
        """
        Generates a grayscale image with a centered Gaussian
        @param sigma: float - Gaussian sigma
        @return: np.array - Gaussian map
        """
        gaussmap = np.zeros((self.boxsize, self.boxsize, 1))
        for x in range(self.boxsize):
            for y in range(self.boxsize):
                dist_sq = (x - self.boxsize / 2) * (x - self.boxsize / 2) \
                          + (y - self.boxsize / 2) * (y - self.boxsize / 2)
                exponent = dist_sq / 2.0 / sigma / sigma
                gaussmap[y, x, :] = math.exp(-exponent)

        return gaussmap

    def crop_humans(self, coords):
        """
        Crop each human in the image.
        @param coords: np.array - human coords
        @return: np.array - cropped humans
        """
        num_people = coords.shape[0]
        boxes = np.ones((num_people, self.boxsize, self.boxsize, 3)) * 128

        pad_h = np.ones((self.im.shape[1] + self.boxsize,
                         self.boxsize / 2, 3)) * 128
        pad_v = np.ones((self.boxsize / 2, self.im.shape[2], 3)) * 128
        im_human = np.vstack((pad_v, np.squeeze(self.im), pad_v))
        im_human = np.hstack((pad_h, im_human, pad_h))
        for i in range(num_people):
            y = coords[i][1] + self.boxsize / 2
            x = coords[i][0] + self.boxsize / 2
            boxes[i, :, :, :] = im_human[
                                y - self.boxsize / 2: y + self.boxsize / 2,
                                x - self.boxsize / 2: x + self.boxsize / 2]
        return boxes
