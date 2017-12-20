#!/usr/bin/env python

"""
caffe_cv.py: Script for testing CPMs original implementation with
Caffe and OpenCV captured video. Based on @shihenw code:
https://github.com/shihenw/convolutional-pose-machines-release/blob/master/testing/python/demo.ipynb
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/12/05"

# Avoids verbosity when loading Caffe model
import os

os.environ['GLOG_minloglevel'] = '2'

import caffe
import cv2
import caffe_cpm as cpm

if __name__ == '__main__':
    caffe.set_mode_cpu()

    model, deploy_models = cpm.load_model()

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        pose_coords, im_predicted = cpm.predict(model, deploy_models, frame, 0)

        # Display the resulting frame
        cv2.imshow("Human pose", im_predicted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
