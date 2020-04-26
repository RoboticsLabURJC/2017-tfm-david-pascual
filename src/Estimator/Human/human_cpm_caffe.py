#!/usr/bin/env python

"""
human_caffe.py: Module for detecting persons with Caffe.
Based on @shihenw code:
https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/python/demo.ipynb
"""

import os

# Avoids verbosity when loading Caffe model
os.environ["GLOG_minloglevel"] = "2"

import caffe
import cv2
import numpy as np

from human import Human

__author__ = "David Pascual Hernandez"
__date__ = "2018/05/22"


def map_resize(new_shape, heatmap):
    # Resizes the output back to the size of the test image
    scale_y = new_shape[0] / float(heatmap.shape[0])
    scale_x = new_shape[1] / float(heatmap.shape[1])
    map_resized = cv2.resize(heatmap, None, fx=scale_x, fy=scale_y,
                             interpolation=cv2.INTER_CUBIC)

    return map_resized


class HumanDetector(Human):
    """
    Class for person detection.
    """

    def __init__(self, model, boxsize):
        """
        Class constructor.
        @param model: caffe models
        @param weights: caffe models weights
        """
        Human.__init__(self, boxsize)

        # Reshapes the models input accordingly
        self.model, self.weights = model
        self.net = None

    def init_net(self):
        caffe.set_mode_gpu()
        self.net = caffe.Net(self.model, self.weights, caffe.TEST)

    def detect(self):
        """
        Detects people in the image.
        @param im: np.array - input image
        @return: np.array - heatmap
        """
        if not self.net:
            self.init_net()

        # Reshapes and normalizes the input image
        im = np.float32(self.im[:, :, :, np.newaxis])
        im = np.transpose(im, (3, 2, 0, 1)) / 256 - 0.5
        self.net.blobs['image'].reshape(*im.shape)

        # Feeds the net
        self.net.blobs['image'].data[...] = im

        # Person detection
        output_blobs = self.net.forward()

        self.heatmap = np.squeeze(self.net.blobs[output_blobs.keys()[0]].data)
        self.heatmap = map_resize(self.im.shape, self.heatmap)

if __name__ == "__main__":
    human_model = ["/home/dpascualhe/repos/2017-tfm-david-pascual/src/Estimator/Human/models/caffe/pose_deploy_copy_4sg_resize.prototxt",
                   "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Estimator/Human/models/caffe/pose_iter_70000.caffemodel"]
    boxsize = 256

    hd = HumanDetector(human_model, boxsize)
    hd.init_net()

    im = cv2.imread("/home/dpascualhe/repos/2017-tfm-david-pascual/src/Estimator/Samples/nadal.png")
    human_bbox = hd.get_bboxes(im)[0]

    (ux, uy), (lx, ly) = human_bbox
    print(human_bbox)
    # Get scale
    scale = float(boxsize) / im.shape[0]

    # Get center
    cx, cy = (int((ux + lx) * scale / 2), int((uy + ly) * scale / 2))

    im_drawn = cv2.rectangle(im.copy(), (ux, uy), (lx, ly), color=(255, 0, 0), thickness=5)

    def crop_human(sample, c, s, bsize):
        """
        Crop human in the image depending on subject center and scale.
        @param sample: np.array - input image
        @param c: list - approx. human center
        @param s: float - approx. human scale wrt 200px
        @param bsize: int - boxsize
        @return: np.array - cropped human
        """
        cx, cy = c

        # Resize image and center according to given scale
        im_resized = cv2.resize(sample, None, fx=s, fy=s)

        h, w, d = im_resized.shape

        pad_up = int(bsize / 2 - cy)
        pad_down = int(bsize / 2 - (h - cy))
        pad_left = int(bsize / 2 - cx)
        pad_right = int(bsize / 2 - (w - cx))

        # Apply padding or crop image as needed
        if pad_up > 0:
            pad = np.ones((pad_up, w, d), np.uint8) * 128
            im_resized = np.vstack((pad, im_resized))
        else:
            im_resized = im_resized[-pad_up:, :, :]
        h, w, d = im_resized.shape

        if pad_down > 0:
            pad = np.ones((pad_down, w, d), np.uint8) * 128
            im_resized = np.vstack((im_resized, pad))
        else:
            im_resized = im_resized[:h + pad_down, :, :]
        h, w, d = im_resized.shape

        if pad_left > 0:
            pad = np.ones((h, pad_left, d), np.uint8) * 128
            im_resized = np.hstack((pad, im_resized))
        else:
            im_resized = im_resized[:, -pad_left:, :]
        h, w, d = im_resized.shape

        if pad_right > 0:
            pad = np.ones((h, pad_right, d), np.uint8) * 128
            im_resized = np.hstack((im_resized, pad))
        else:
            im_resized = im_resized[:, :w + pad_right, :]

        return im_resized

    im_human = crop_human(im, (cx, cy), scale, boxsize)

    from matplotlib import pyplot as plt
    plt.figure()
    plt.subplot(121), plt.imshow(im_drawn[:, :, ::-1])
    plt.subplot(122), plt.imshow(im_human[:, :, ::-1])
    plt.show()
