#!/usr/bin/env python

"""
humanpose.py: Receive images from live video and estimate human pose.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import signal
import sys
import yaml
from PyQt5 import QtWidgets

from Camera.camera import Camera
from Camera.threadcamera import ThreadCamera
from Estimator.estimator import Estimator
from Estimator.threadestimator import ThreadEstimator
from GUI.gui import GUI
from GUI.threadgui import ThreadGUI
from Viz3D.viz3d import Viz3D

signal.signal(signal.SIGINT, signal.SIG_DFL)

def selectVideoSource(cfg):
    """
    @param cfg: configuration
    @return cam: selected camera
    @raise SystemExit in case of unsupported video source
    """
    source = cfg['HumanPose']['Source']
    if source.lower() == 'local':
        from Camera.local_camera import Camera
        cam_idx = cfg['HumanPose']['Local']['DeviceNo']
        print('Chosen source: local camera (index %d)' % (cam_idx))
        cam = Camera(cam_idx)
    elif source.lower() == 'video':
        from Camera.local_video import Camera
        video_path = cfg['HumanPose']['Video']['Path']
        print('Chosen source: local video (%s)' % (video_path))
        cam = Camera(video_path)
    elif source.lower() == 'stream':
        # comm already prints the source technology (ICE/ROS)
        import comm
        import config
        cfg = config.load(sys.argv[1])
        jdrc = comm.init(cfg, 'HumanPose.Stream')
        from Camera.stream_camera import Camera
        cam = Camera(jdrc)
    else:
        raise SystemExit(('%s not supported! Supported source: Local, Video, Stream') % (source))

    return cam

def readConfig():
    try:
        with open(sys.argv[1], 'r') as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        raise SystemExit('Error: Cannot read/parse YML file. Check YAML syntax.')
    except:
        raise SystemExit('\n\tUsage: python2 objectdetector.py objectdetector.yml\n')

if __name__ == "__main__":
    # Init objects
    app = QtWidgets.QApplication(sys.argv)

    data = readConfig()
    cam = selectVideoSource(data)

    viz3d = Viz3D()
    window = GUI(cam)
    window.show()
    estimator = Estimator(cam, viz3d, window, data["Estimator"])

    # Threading camera
    t_cam = ThreadCamera(cam)
    t_cam.start()

    # Threading estimator
    t_estimator = ThreadEstimator(estimator)
    t_estimator.start()

    # Threading GUI
    t_gui = ThreadGUI(window)
    t_gui.start()

    sys.exit(app.exec_())
