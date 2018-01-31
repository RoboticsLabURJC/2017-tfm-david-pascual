#!/usr/bin/env python

"""
pose_estimator.py: Module for estimating human pose with TensorFlow.
Based on @psycharm code:
https://github.com/psycharo/cpm/blob/master/example.ipynb
"""

import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

__author__ = "David Pascual Hernandez"
__date__ = "2018/01/18"


def inference_pose(image, center_map):
    """
    Human detection models.
    @param image: np.array - human image
    @param center_map: np.array - gaussian map
    @return: TensorFlow models
    """
    with tf.variable_scope('PoseNet'):
        pool_center_lower = layers.avg_pool2d(center_map, 9, 8, padding='VALID')
        conv1_stage1 = layers.conv2d(image, 128, 9, 1, activation_fn=None,
                                     scope='conv1_stage1')
        conv1_stage1 = tf.nn.relu(conv1_stage1)
        pool1_stage1 = layers.max_pool2d(conv1_stage1, 3, 2)
        conv2_stage1 = layers.conv2d(pool1_stage1, 128, 9, 1,
                                     activation_fn=None, scope='conv2_stage1')
        conv2_stage1 = tf.nn.relu(conv2_stage1)
        pool2_stage1 = layers.max_pool2d(conv2_stage1, 3, 2)
        conv3_stage1 = layers.conv2d(pool2_stage1, 128, 9, 1,
                                     activation_fn=None, scope='conv3_stage1')
        conv3_stage1 = tf.nn.relu(conv3_stage1)
        pool3_stage1 = layers.max_pool2d(conv3_stage1, 3, 2)
        conv4_stage1 = layers.conv2d(pool3_stage1, 32, 5, 1, activation_fn=None,
                                     scope='conv4_stage1')
        conv4_stage1 = tf.nn.relu(conv4_stage1)
        conv5_stage1 = layers.conv2d(conv4_stage1, 512, 9, 1,
                                     activation_fn=None, scope='conv5_stage1')
        conv5_stage1 = tf.nn.relu(conv5_stage1)
        conv6_stage1 = layers.conv2d(conv5_stage1, 512, 1, 1,
                                     activation_fn=None, scope='conv6_stage1')
        conv6_stage1 = tf.nn.relu(conv6_stage1)
        conv7_stage1 = layers.conv2d(conv6_stage1, 15, 1, 1, activation_fn=None,
                                     scope='conv7_stage1')
        conv1_stage2 = layers.conv2d(image, 128, 9, 1, activation_fn=None,
                                     scope='conv1_stage2')
        conv1_stage2 = tf.nn.relu(conv1_stage2)
        pool1_stage2 = layers.max_pool2d(conv1_stage2, 3, 2)
        conv2_stage2 = layers.conv2d(pool1_stage2, 128, 9, 1,
                                     activation_fn=None, scope='conv2_stage2')
        conv2_stage2 = tf.nn.relu(conv2_stage2)
        pool2_stage2 = layers.max_pool2d(conv2_stage2, 3, 2)
        conv3_stage2 = layers.conv2d(pool2_stage2, 128, 9, 1,
                                     activation_fn=None, scope='conv3_stage2')
        conv3_stage2 = tf.nn.relu(conv3_stage2)
        pool3_stage2 = layers.max_pool2d(conv3_stage2, 3, 2)
        conv4_stage2 = layers.conv2d(pool3_stage2, 32, 5, 1, activation_fn=None,
                                     scope='conv4_stage2')
        conv4_stage2 = tf.nn.relu(conv4_stage2)
        concat_stage2 = tf.concat(axis=3, values=[conv4_stage2, conv7_stage1,
                                                  pool_center_lower])
        Mconv1_stage2 = layers.conv2d(concat_stage2, 128, 11, 1,
                                      activation_fn=None, scope='Mconv1_stage2')
        Mconv1_stage2 = tf.nn.relu(Mconv1_stage2)
        Mconv2_stage2 = layers.conv2d(Mconv1_stage2, 128, 11, 1,
                                      activation_fn=None, scope='Mconv2_stage2')
        Mconv2_stage2 = tf.nn.relu(Mconv2_stage2)
        Mconv3_stage2 = layers.conv2d(Mconv2_stage2, 128, 11, 1,
                                      activation_fn=None, scope='Mconv3_stage2')
        Mconv3_stage2 = tf.nn.relu(Mconv3_stage2)
        Mconv4_stage2 = layers.conv2d(Mconv3_stage2, 128, 1, 1,
                                      activation_fn=None, scope='Mconv4_stage2')
        Mconv4_stage2 = tf.nn.relu(Mconv4_stage2)
        Mconv5_stage2 = layers.conv2d(Mconv4_stage2, 15, 1, 1,
                                      activation_fn=None, scope='Mconv5_stage2')
        conv1_stage3 = layers.conv2d(pool3_stage2, 32, 5, 1, activation_fn=None,
                                     scope='conv1_stage3')
        conv1_stage3 = tf.nn.relu(conv1_stage3)
        concat_stage3 = tf.concat(axis=3, values=[conv1_stage3, Mconv5_stage2,
                                                  pool_center_lower])
        Mconv1_stage3 = layers.conv2d(concat_stage3, 128, 11, 1,
                                      activation_fn=None, scope='Mconv1_stage3')
        Mconv1_stage3 = tf.nn.relu(Mconv1_stage3)
        Mconv2_stage3 = layers.conv2d(Mconv1_stage3, 128, 11, 1,
                                      activation_fn=None, scope='Mconv2_stage3')
        Mconv2_stage3 = tf.nn.relu(Mconv2_stage3)
        Mconv3_stage3 = layers.conv2d(Mconv2_stage3, 128, 11, 1,
                                      activation_fn=None, scope='Mconv3_stage3')
        Mconv3_stage3 = tf.nn.relu(Mconv3_stage3)
        Mconv4_stage3 = layers.conv2d(Mconv3_stage3, 128, 1, 1,
                                      activation_fn=None, scope='Mconv4_stage3')
        Mconv4_stage3 = tf.nn.relu(Mconv4_stage3)
        Mconv5_stage3 = layers.conv2d(Mconv4_stage3, 15, 1, 1,
                                      activation_fn=None, scope='Mconv5_stage3')
        conv1_stage4 = layers.conv2d(pool3_stage2, 32, 5, 1, activation_fn=None,
                                     scope='conv1_stage4')
        conv1_stage4 = tf.nn.relu(conv1_stage4)
        concat_stage4 = tf.concat(axis=3, values=[conv1_stage4, Mconv5_stage3,
                                                  pool_center_lower])
        Mconv1_stage4 = layers.conv2d(concat_stage4, 128, 11, 1,
                                      activation_fn=None, scope='Mconv1_stage4')
        Mconv1_stage4 = tf.nn.relu(Mconv1_stage4)
        Mconv2_stage4 = layers.conv2d(Mconv1_stage4, 128, 11, 1,
                                      activation_fn=None, scope='Mconv2_stage4')
        Mconv2_stage4 = tf.nn.relu(Mconv2_stage4)
        Mconv3_stage4 = layers.conv2d(Mconv2_stage4, 128, 11, 1,
                                      activation_fn=None, scope='Mconv3_stage4')
        Mconv3_stage4 = tf.nn.relu(Mconv3_stage4)
        Mconv4_stage4 = layers.conv2d(Mconv3_stage4, 128, 1, 1,
                                      activation_fn=None, scope='Mconv4_stage4')
        Mconv4_stage4 = tf.nn.relu(Mconv4_stage4)
        Mconv5_stage4 = layers.conv2d(Mconv4_stage4, 15, 1, 1,
                                      activation_fn=None, scope='Mconv5_stage4')
        conv1_stage5 = layers.conv2d(pool3_stage2, 32, 5, 1, activation_fn=None,
                                     scope='conv1_stage5')
        conv1_stage5 = tf.nn.relu(conv1_stage5)
        concat_stage5 = tf.concat(axis=3, values=[conv1_stage5, Mconv5_stage4,
                                                  pool_center_lower])
        Mconv1_stage5 = layers.conv2d(concat_stage5, 128, 11, 1,
                                      activation_fn=None, scope='Mconv1_stage5')
        Mconv1_stage5 = tf.nn.relu(Mconv1_stage5)
        Mconv2_stage5 = layers.conv2d(Mconv1_stage5, 128, 11, 1,
                                      activation_fn=None, scope='Mconv2_stage5')
        Mconv2_stage5 = tf.nn.relu(Mconv2_stage5)
        Mconv3_stage5 = layers.conv2d(Mconv2_stage5, 128, 11, 1,
                                      activation_fn=None, scope='Mconv3_stage5')
        Mconv3_stage5 = tf.nn.relu(Mconv3_stage5)
        Mconv4_stage5 = layers.conv2d(Mconv3_stage5, 128, 1, 1,
                                      activation_fn=None, scope='Mconv4_stage5')
        Mconv4_stage5 = tf.nn.relu(Mconv4_stage5)
        Mconv5_stage5 = layers.conv2d(Mconv4_stage5, 15, 1, 1,
                                      activation_fn=None, scope='Mconv5_stage5')
        conv1_stage6 = layers.conv2d(pool3_stage2, 32, 5, 1, activation_fn=None,
                                     scope='conv1_stage6')
        conv1_stage6 = tf.nn.relu(conv1_stage6)
        concat_stage6 = tf.concat(axis=3, values=[conv1_stage6, Mconv5_stage5,
                                                  pool_center_lower])
        Mconv1_stage6 = layers.conv2d(concat_stage6, 128, 11, 1,
                                      activation_fn=None, scope='Mconv1_stage6')
        Mconv1_stage6 = tf.nn.relu(Mconv1_stage6)
        Mconv2_stage6 = layers.conv2d(Mconv1_stage6, 128, 11, 1,
                                      activation_fn=None, scope='Mconv2_stage6')
        Mconv2_stage6 = tf.nn.relu(Mconv2_stage6)
        Mconv3_stage6 = layers.conv2d(Mconv2_stage6, 128, 11, 1,
                                      activation_fn=None, scope='Mconv3_stage6')
        Mconv3_stage6 = tf.nn.relu(Mconv3_stage6)
        Mconv4_stage6 = layers.conv2d(Mconv3_stage6, 128, 1, 1,
                                      activation_fn=None, scope='Mconv4_stage6')
        Mconv4_stage6 = tf.nn.relu(Mconv4_stage6)
        Mconv5_stage6 = layers.conv2d(Mconv4_stage6, 15, 1, 1,
                                      activation_fn=None, scope='Mconv5_stage6')
    return Mconv5_stage6


class PoseEstimator:
    def __init__(self, humans, config, model_path, boxsize):
        """
        Class for human detection.
        @param humans: human images and gaussian maps
        @param config: TensorFlow configuration
        @param model_path: human detection models path
        @param boxsize: int - boxsize
        """
        self.config = config
        self.model_path = model_path
        self.boxsize = boxsize

        n, h, w = humans.shape[:3]
        with tf.variable_scope('CPM'):
            # input dims for the pose network
            im_human = tf.placeholder(tf.float32, [n, h, w, 3])
            map_human = tf.placeholder(tf.float32, [n, h, w, 1])
            self.map_pose = inference_pose(im_human, map_human)

        ims = humans[:, :, :, :3]
        maps = humans[:, :, :, 3][:, :, :, np.newaxis]
        self.feed_dict = {im_human: ims, map_human: maps}

        self.sess = tf.Session(config=self.config)
        self.set_model()

    def set_model(self):
        """
        Get the models ready for inference.
        """
        model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  'CPM/PoseNet')
        restorer = tf.train.Saver(model)

        restorer.restore(self.sess, self.model_path)

    def get_map(self):
        """
        Get joint heatmaps.
        @return: np.array - joint heatmaps
        """

        start_time = time.time()
        map_pose = self.sess.run(self.map_pose, self.feed_dict)
        print("Pose estimated! (%.2f ms)" % (1000 * (time.time() - start_time)))

        return self.resize_maps(map_pose)

    def resize_maps(self, maps):
        """
        Resize pose maps back to boxsize
        :param maps: np.array - each joint of each human maps
        :return: np.array: resized maps
        """
        scalex = self.boxsize / float(maps.shape[1])
        scaley = self.boxsize / float(maps.shape[2])

        maps_full_joints = []
        for map_joints in enumerate(maps):

            map_joints = map_joints[1]

            map_full_joints = []
            for i in range(map_joints.shape[2]):
                map_joint = map_joints[:, :, i]
                map_full_joint = cv2.resize(map_joint, None,
                                            fx=scalex, fy=scaley,
                                            interpolation=cv2.INTER_CUBIC)
                map_full_joints.append(np.array(map_full_joint))

            maps_full_joints.append(np.array(map_full_joints))

        return np.squeeze(np.array(maps_full_joints))

    def get_joints(self, joint_maps, person_coords, r):
        """
        Get joint coordinates and resize them given a heatmap.
        @param joint_map: np.array - heatmap
        @param person_coords: np.array - person center coordinates
        @param r: float - resize rate
        @return: joint coordinates
        """
        # Get coordinates
        joint_coords = []
        for joint_map in joint_maps:
            y, x = np.unravel_index(np.argmax(joint_map), (self.boxsize,
                                                           self.boxsize))

            # Back to full coordinates
            x = int((x + person_coords[0] - (self.boxsize / 2)) / r)
            y = int((y + person_coords[1] - (self.boxsize / 2)) / r)
            joint_coords.append([x, y])

        return np.array(joint_coords)

    def draw_limbs(self, im, joints, data):
        """
        Draw estimated limb positions over the image.
        @param im: np.array - image
        @param joints: np.array - joints estimated for every human
        @param data: parsed YAML config. file
        @return: np.array - image with limbs drawn
        """
        limbs = np.array(data["limbs"]).reshape((-1, 2)) - 1
        colors = data["colors"]

        for i, (p, q) in enumerate(limbs):
            px, py = joints[p]
            qx, qy = joints[q]
            cv2.line(im, (px, py), (qx, qy), colors[i], 2)

        return im
