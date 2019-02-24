#!/usr/bin/env bash

# TensorFlow models
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bw6m_66JSYLldnV0a1JUbFpMNVE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bw6m_66JSYLldnV0a1JUbFpMNVE" -O Estimator/Human/models/tf/person_net.ckpt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bw6m_66JSYLlTVNlcHRkeEVaeE0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bw6m_66JSYLlTVNlcHRkeEVaeE0" -O Estimator/Pose/models/tf/pose_net.ckpt && rm -rf /tmp/cookies.txt

# Caffe models
wget -nc --directory-prefix=Estimator/Human/models/caffe/   http://posefs1.perception.cs.cmu.edu/CPM/caffe_model_github/model/_trained_person_MPI/pose_iter_70000.caffemodel
wget -nc --directory-prefix=Estimator/Pose/models/caffe/    http://posefs1.perception.cs.cmu.edu/CPM/caffe_model_github/model/_trained_MPI/pose_iter_320000.caffemodel

# @naxvm models
wget -nc --directory-prefix=Estimator/Human/models/naxvm/   http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
tar -xvzf Estimator/Human/models/naxvm/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz && rm ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
