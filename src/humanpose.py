#!/usr/bin/env python

"""
humanpose.py: Receive images from live video and estimate human pose.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import Ice
import signal
import sys
import torch  # import here to avoid bugs
import traceback
import yaml
from PyQt5 import QtWidgets

from Camera.threadcamera import ThreadCamera
from Estimator.estimator import Estimator
from Estimator.threadestimator import ThreadEstimator
from GUI.gui import GUI
from GUI.gui_3d import GUI3D
from GUI.threadgui import ThreadGUI
from Viz.viz3d import Viz3D

signal.signal(signal.SIGINT, signal.SIG_DFL)


def readConfig():
    try:
        with open(sys.argv[1], 'r') as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        raise SystemExit(
            'Error: Cannot read/parse YML file. Check YAML syntax.')
    except:
        raise SystemExit('\n\tUsage: python2 humanpose.py humanpose.yml\n')


def selectVideoSource(cfg):
    """
    @param cfg: configuration
    @return cam: selected camera
    @raise SystemExit in case of unsupported video source
    """
    cam, cam_depth = (None, None)

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

    elif source.lower() == 'jder':
        # comm already prints the source technology (ICE/ROS)
        import comm
        import config
        from Camera.jder_camera import Camera

        cfg = config.load(sys.argv[1])
        jdrc = comm.init(cfg, 'HumanPose.JdeR')

        try:
            prx_rgb = jdrc.getCameraClient('HumanPose.JdeR.CameraRGB')
            cam = Camera(prx_rgb)
        except:
            traceback.print_exc()
            raise SystemExit("No RGB camera found!")

        try:
            prx_depth = jdrc.getCameraClient('HumanPose.JdeR.CameraDEPTH')
            cam_depth = Camera(prx_depth)
        except:
            traceback.print_exc()
            print("No depth camera found!")

    elif source == "ros":
        from Camera.ros_camera import Camera

        rgb = cfg['HumanPose']['ROS']['CameraRGB']
        try:
            cam = Camera(rgb['Topic'], rgb['Format'])
        except:
            traceback.print_exc()
            raise SystemExit("No RGB camera found!")

        depth = cfg['HumanPose']['ROS']['CameraDEPTH']
        try:
            cam_depth = Camera(depth["Topic"], depth["Format"], depth["fx"], depth["cx"], depth["fy"], depth["cy"])
        except:
            traceback.print_exc()
            print("No depth camera found!")

    else:
        msg = '%s not supported! Available sources: local, video, jder' % \
              source
        raise SystemExit(msg)

    return cam, cam_depth

def init_viz():
    object = Viz3D()

    endpoint = "default -h localhost -p 9957:ws -h localhost -p 11000"
    print("Connect: " + endpoint)

    id = Ice.InitializationData()
    ic = Ice.initialize(None, id)

    adapter = ic.createObjectAdapterWithEndpoints("3DVizA", endpoint)
    adapter.add(object, ic.stringToIdentity("3DViz"))
    adapter.activate()

    return object


if __name__ == "__main__":
    # Init objects
    app = QtWidgets.QApplication(sys.argv)

    data = readConfig()
    cam, cam_depth = selectVideoSource(data)

    viz3d = None
    if cam_depth:
        viz3d = init_viz()
        window = GUI3D(cam, cam_depth)
    else:
        window = GUI(cam)
    window.show()

    estimator = Estimator(cam, cam_depth, viz3d, window, data["Estimator"])

    # Threading camera
    t_cam = ThreadCamera(cam)
    t_cam.setDaemon(True)
    t_cam.start()
    t_cam_depth = ThreadCamera(cam_depth)
    t_cam_depth.setDaemon(True)
    t_cam_depth.start()

    # Threading estimator
    t_estimator = ThreadEstimator(estimator)
    t_estimator.setDaemon(True)
    t_estimator.start()

    # Threading GUI
    t_gui = ThreadGUI(window)
    t_gui.setDaemon(True)
    t_gui.start()

    sys.exit(app.exec_())
