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

import traceback
import threading
import numpy as np

class Camera:

    def __init__ (self, jdrc):
        ''' Camera class gets images from live video and transform them
        in order to detect objects in the image.
        '''
        self.cam_rgb = False
        self.cam_depth = False

        self.lock = threading.Lock()
        # noinspection PyBroadException
        try:
            self.cam_rgb = jdrc.getCameraClient('HumanPose.Stream.CameraRGB')
            if self.cam_rgb.hasproxy():
                self.im_rgb = self.cam_rgb.getImage()
                self.im_height = self.im_rgb.height
                self.im_width = self.im_rgb.width
                print("RGB camera succesfully connected!")
            else:
                raise SystemExit
        except:
            traceback.print_exc()
            exit()

        # noinspection PyBroadException
        try:
            self.cam_depth = jdrc.getCameraClient('HumanPose.Stream.CameraDEPTH')
            if self.cam_depth.hasproxy():
                self.im_depth = self.cam_depth.getImage()
                print("Depth camera succesfully connected!")
            else:
                raise SystemExit
        except:
            print("Depth camera not found!")


    def get_image(self):
        ''' Gets the image from the webcam and returns it. '''
        im_rgb = np.zeros((self.im_height, self.im_width, 3))
        im_depth = np.zeros((self.im_height, self.im_width, 1))

        if self.cam_rgb:
            im_rgb = np.frombuffer(self.im_rgb.data, dtype=np.uint8)
            im_rgb = np.reshape(im_rgb, (self.im_height, self.im_width, 3))
        if self.cam_depth:
            im_depth = np.frombuffer(self.im_depth.data, dtype=np.uint8)
            im_depth = np.reshape(im_depth, (self.im_height, self.im_width, 3))
        

        return im_rgb, im_depth

    def update(self):
        ''' Updates the camera with an incoming stream every time the thread changes. '''
        if self.cam_rgb:
            self.lock.acquire()
            self.im_rgb = self.cam_rgb.getImage()
            self.lock.release()
        
        if self.cam_depth:
            self.lock.acquire()
            self.im_depth = self.cam_depth.getImage()
            self.lock.release()
