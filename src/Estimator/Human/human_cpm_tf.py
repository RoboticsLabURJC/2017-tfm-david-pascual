#!/usr/bin/env python

"""
human_detector.py: Module for human detection with TensorFlow.
Based on @psycharm code:
https://github.com/psycharo/cpm/blob/master/example.ipynb
"""

__author__ = "David Pascual Hernandez"
__date__ = "2017/05/22"

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from human import Human


def inference_human(image):
    """
    Human detection models.
    @param image: np.array - human image
    @return: TensorFlow models
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


class HumanDetector(Human):
    """
    Class for person detection.
    """

    def __init__(self, model, boxsize):
        """
        Class constructor.
        @param model: tf models
        @param weights: tf models weights
        """
        Human.__init__(self, boxsize)

        self.config = tf.ConfigProto(device_count={"GPU": 1},
                                     allow_soft_placement=True,
                                     log_device_placement=False)
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.5

        self.sess = None

        self.weights = model

        self.first_detection = True
        self.image_in = None
        self.map_human_large = None

    def init_net(self):
        """
        Get the models ready for inference.
        """
        h, w = self.im.shape[:2]
        with tf.variable_scope('CPM'):
            # Input dims for the human models
            self.image_in = tf.placeholder(tf.float32, [1, h, w, 3])

            map_human = inference_human(self.image_in)
            self.map_human_large = tf.image.resize_images(map_human, [h, w])

        self.sess = tf.Session(config=self.config)

        model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  'CPM/PersonNet')

        restorer = tf.train.Saver(model)
        restorer.restore(self.sess, self.weights)

    def detect(self):
        """
        Detects people in the image.
        @param im: np.array - input image
        @return: np.array - heatmap
        """
        if self.first_detection:
            self.init_net()
            self.first_detection = False

        tf.reset_default_graph()

        im = self.im[np.newaxis] / 255.0 - 0.5
        self.heatmap = self.sess.run(self.map_human_large, {self.image_in: im})
        self.heatmap = np.squeeze(self.heatmap, 0)
