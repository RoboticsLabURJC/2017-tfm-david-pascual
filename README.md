# 2017-tfm-david-pascual

**Project Name:** Convolutional Neural Networks for Human Pose Estimation

**Author:** David Pascual Hernández [d.pascualhe@gmail.com]

**Academic Year:** 2017/2018

**Degree:** Computer Vision Master (URJC)

**Mediawiki:** http://jderobot.org/Dpascual-tfm

**Tags:** Deep Learning, Tensorflow, Convolutional Pose Machines, Human Pose Estimation

**State:** Developing 

[Convolutional Pose Machines (CPMs)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/app/S20-08.pdf) are multi-stage
convolutional neural networks for estimating articulated poses, like human pose estimation. In this repo, a tool for
live testing CPMs is provided. We currently host two implementations based on different frameworks:
* [Caffe implementation](https://github.com/shihenw/convolutional-pose-machines-release) (official repo).
* [TensorFlow implementation](https://github.com/psycharo/cpm).

## Dependencies
* JdeRobot ([installation guide](http://jderobot.org/Installation))
* TensorFlow ([installation guide](https://www.tensorflow.org/install/install_linux))
* Caffe ([recommended installation guide](https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-CPU-Only/))

Other dependencies like Numpy, PyQT or OpenCV should be automatically installed along with the previous ones. By the way, 
if you install TensorFlow and/or Caffe with GPU support, the code provided in this repo will take advantage of it to get 
closer to real-time estimation (we're almost there).

## Usage
In order to test CPMs with live video feed, we have built <code>humanpose</code> component within the framework of
[JdeRobot](http://jderobot.org/), a middleware that provides several tools and drivers for robotics and computer vision tasks.

Once all the dependencies have been installed and the repo has been cloned, download the trained Caffe and TensorFlow models:
<pre>
chmod +x get_models.sh
./get_models.sh
</pre>

Then you must open a terminal and run:
<pre>
cameraserver cameraserver.cfg
</pre>
This will launch <code>cameraserver</code>, which will serve live video from your webcam (or any other source of video).
Now, from another terminal run:
<pre>
python humanpose.py humanpose.yml
</pre>

This will launch a GUI where live video and estimated pose are shown and you should see something like this:
![Alt text](http://jderobot.org/store/dpascual/uploads/images/tfm/humapose.png)

If you want to change the framework that will be used for live estimation or any other parameter related with the 
CPMs, you must edit the YAML configuration file.


 

More info: http://jderobot.org/Dpascual-tfm
