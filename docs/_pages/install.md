---
permalink: /install/

title: "Installation and use"

sidebar:
  nav: "docs"
---

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

Once all the dependencies have been installed and the repo has been cloned, download the trained Caffe and TensorFlow models (provided by their authors):
<pre>
chmod +x get_models.sh
./get_models.sh
</pre>

You can choose different video sources to feed the CPM (ROS, JdeRobot, local webcam, video file). The video source, as
well as other settings can be modified from the <code>humanpose.yml</code> file. When you have set your desired
parameters, from another terminal run:
<pre>
python humanpose.py humanpose.yml
</pre>

This will launch a GUI where live video and estimated poses are shown.

### 3D Visualization
RGBD camera can be used in order to get the pose 3D coordinates. 3DVizWeb, developed within JdeRobot, allow us to project
the estimated joints and limbs in 3D. If you want to use this feature, you must follow the instructions available
[here](https://github.com/RoboticsURJC-students/2017-tfm-david-pascual/tree/master/src/Viz/3DVizWeb) and run: 
<pre>
cd Viz/3DVizWeb/
npm start
</pre>

right after launching <code>humanpose</code>.

## Demo
Click in the image below to watch a real-time demo:
[![Watch the video](https://img.youtube.com/vi/926rJOixlFA/maxresdefault.jpg)](https://youtu.be/926rJOixlFA)

The input data for this demo is available as a
[rosbag file](https://mega.nz/#!4U8nXAib!1zbaeYGGraTqdUVbbQneG28PA50gr6U3WeqIKzoIup0), containing
the registered depth map and color images.
