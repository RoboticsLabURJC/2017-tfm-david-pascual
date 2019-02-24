#!/usr/bin/env python

import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

    def __init__(self, topic, im_format):
        self.im_format = im_format

        topics_list = [t for ts in rospy.get_published_topics() for t in ts]
        if topic not in topics_list:
            raise Exception("Topic %s not found!" % topic)

        self.cv_image = []
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image,
                                          self.callback)

    def callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, self.im_format)

class Camera:
    def __init__(self, topic, im_format, fx=0, cx=0, fy=0, cy=0):
        print(topic, im_format)
        self.ic = image_converter(topic, im_format)
        rospy.init_node('converter', anonymous=True)

        # Get first frame when ready to set width/height attributes
        while not len(self.ic.cv_image):
            pass
        im = self.ic.cv_image
        self.im_width = im.shape[1]
        self.im_height = im.shape[0]

        self.calib_data = {}
        self.calib_data["fx"] = fx
        self.calib_data["cx"] = cx
        self.calib_data["fy"] = fy
        self.calib_data["cy"] = cy

    def get_image(self):
        return self.ic.cv_image

    def update(self):
        # do nothing, image is updated by callback
        pass
