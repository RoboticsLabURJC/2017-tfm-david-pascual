#
# Created on Jan, 2018
#
# @author: naxvm
#
# Class which abstracts a Camera from a proxy (created by ICE/ROS),
# and provides the methods to keep it constantly updated. Also, delivers it
# to the neural network, which returns returns the same image with the
# detected classes and scores, and the bounding boxes drawn on it.
#

import threading
import numpy as np

class Camera:

    def __init__ (self, prx):
        ''' Camera class gets images from live video and transform them
        in order to detect objects in the image.
        '''
        self.cam = prx
        self.lock = threading.Lock()

        if self.cam.hasproxy():
            self.im = self.cam.getImage()
            self.im_height = self.im.height
            self.im_width = self.im.width
            print("Camera succesfully connected!")
            print("\tImage size: (%d, %d)" % (self.im_width, self.im_height))
        else:
            raise Exception("No proxy!")


    def get_image(self):
        ''' Gets the image from the webcam and returns it. '''
        im = np.zeros((self.im_height, self.im_width, 3))

        if self.cam:
            im = np.frombuffer(self.im.data, dtype=np.uint8)
            im = np.reshape(im, (self.im_height, self.im_width, 3))

        return im

    def update(self):
        ''' Updates the camera with an incoming stream every time the thread changes. '''
        if self.cam:
            self.lock.acquire()
            self.im = self.cam.getImage()
            self.im_height = self.im.height
            self.im_width = self.im.width
            self.lock.release()
