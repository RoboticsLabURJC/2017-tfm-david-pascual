#!/usr/bin/env python
from __future__ import print_function

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

    def __init__(self, topic, im_format):
        self.im_format = im_format

        topics_list = [t for ts in rospy.get_published_topics() for t in ts]
        if topic not in topics_list:
            raise Exception("Topic %s not found!" % topic)

        self.cv_image = None
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image,
                                          self.callback)

    def callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, self.im_format)

class Camera:
    def __init__(self, topic, im_format):
        self.ic = image_converter(topic, im_format)
        rospy.init_node('image_converter', anonymous=True)

    def get_image(self):
        return self.ic.cv_image

    def update(self):
        rospy.spin()
