# 2017-tfm-david-pascual

**Project Name:** Convolutional Neural Networks for Human Pose Estimation

**Author:** David Pascual Hernández [d.pascualhe@gmail.com]

**Academic Year:** 2017/2018

**Degree:** Computer Vision Master (URJC)

**Mediawiki:** http://jderobot.org/Dpascual-tfm

**Tags:** Deep Learning, Tensorflow, Convolutional Pose Machines, Human Pose Estimation

**State:** Developing 

## Usage
This is an implementation of a human pose estimator based in [Convolutional Pose Machines (CVPR'16)](https://github.com/shihenw/convolutional-pose-machines-release). The original paper and the datasets that have been used to train the model are available in that link. For executing the code in this repo you must first install:
* Caffe ([recommended installation guide](https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-CPU-Only/))
* OpenCV 3 ([recommended installation guide](https://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/)). Please note that if you're planning to install JdeRobot, OpenCV 3 will be automatically installed.

First things first, clone the repo and download the trained Caffe models:
<pre>
git clone https://github.com/RoboticsURJC-students/2017-tfm-david-pascual.git
cd Estimator/Caffe
chmod +x get_models.sh
./get_models.sh
</pre>

Now, you're ready to test the code. There are currently two options:

### Short one
First option is launching <code>cpm_caffe.py</code> to test CPMs implementation and how it works. From the root of the repo run:
<pre>
cd Estimator/Caffe
python cpm_caffe.py path-to-image
</pre>
This will take an image, find human/s and draw estimated pose/s. Additionally, it'll draw some figures during execution to monitor the whole process.

### Long (and cooler) one:
For this one, you must install JdeRobot ([installation guide](http://jderobot.org/Installation)), a middleware for robotics and computer vision which provides several drivers and components for multiple tasks. In this sense, we're currently building <code>humanpose</code>, a JdeRobot component which aims to be a tool for testing different implementations of human pose estimators.
First, you must open a terminal and run:
<pre>
cameraserver cameraserver.cfg
</pre>
This will launch <code>cameraserver</code>, which will serve live video from the webcam.
In another terminal (from the root of the repo) run:
<pre>
python humanpose.py humanpose.cfg
</pre>
This will launch a GUI where live video and estimated pose are shown. Please note that it is under development so it may freeze sometimes.
![Alt text](http://jderobot.org/store/dpascual/uploads/images/tfm/humapose.png)

More info: http://jderobot.org/Dpascual-tfm
