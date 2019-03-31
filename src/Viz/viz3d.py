import jderobot
import math
from random import uniform
from random import randrange

import numpy as np
import pyquaternion as pq

bufferpoints = []
bufferline = []
bufferpose3D = []
id_list = []

obj_list = ["/home/dpascualhe/repos/2017-tfm-david-pascual/src/Viz/3DVizWeb/skeleton/head.obj",
            "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Viz/3DVizWeb/skeleton/torso.obj",
            "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Viz/3DVizWeb/skeleton/arm_right.obj",
            "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Viz/3DVizWeb/skeleton/arm_left.obj",
            "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Viz/3DVizWeb/skeleton/forearm_right.obj",
            "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Viz/3DVizWeb/skeleton/forearm_left.obj",
            "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Viz/3DVizWeb/skeleton/thigh_right.obj",
            "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Viz/3DVizWeb/skeleton/thigh_left.obj",
            "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Viz/3DVizWeb/skeleton/leg_right.obj",
            "/home/dpascualhe/repos/2017-tfm-david-pascual/src/Viz/3DVizWeb/skeleton/leg_left.obj"]

scales = [3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2]

init_pos = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]

init_quat = [pq.Quaternion(axis=[1, 0, 0], degrees=0),
             pq.Quaternion(axis=[1, 0, 0], degrees=0),
             pq.Quaternion(axis=[1, 0, 0], degrees=0),
             pq.Quaternion(axis=[1, 0, 0], degrees=0),
             pq.Quaternion(axis=[1, 0, 0], degrees=0),
             pq.Quaternion(axis=[1, 0, 0], degrees=0),
             pq.Quaternion(axis=[1, 0, 0], degrees=0),
             pq.Quaternion(axis=[1, 0, 0], degrees=0),
             pq.Quaternion(axis=[1, 0, 0], degrees=0),
             pq.Quaternion(axis=[1, 0, 0], degrees=0)]

refresh = True


class Viz3D(jderobot.Visualization):
    def __init__(self):
        self.cont = 0

    def drawPoint(self, point, color=(0, 0, 0), current=None):
        pointJde = jderobot.Point()
        pointJde.x = float(point[0] / 100.0)
        pointJde.y = float(point[1] / 100.0)
        pointJde.z = float(point[2] / 100.0)

        colorJDE = jderobot.Color()
        colorJDE.r = float(color[0])
        colorJDE.g = float(color[1])
        colorJDE.b = float(color[2])
        getbufferPoint(pointJde, colorJDE)

    def drawSegment(self, point_a, point_b, color=(0, 0, 0)):
        pointJde_a = jderobot.Point()
        pointJde_a.x = float(point_a[0] / 100.0)
        pointJde_a.y = float(point_a[1] / 100.0)
        pointJde_a.z = float(point_a[2] / 100.0)

        pointJde_b = jderobot.Point()
        pointJde_b.x = float(point_b[0] / 100.0)
        pointJde_b.y = float(point_b[1] / 100.0)
        pointJde_b.z = float(point_b[2] / 100.0)

        segJde = jderobot.Segment()
        segJde.fromPoint = pointJde_a
        segJde.toPoint = pointJde_b

        colorJDE = jderobot.Color()
        colorJDE.r = float(color[0])
        colorJDE.g = float(color[1])
        colorJDE.b = float(color[2])
        getbufferSegment(segJde, colorJDE, True)

    def drawPose3d(self, obj_id, point, quaternion, h):
        pose3d = jderobot.Pose3DData()
        pose3d.x = float(point[0] / 100.0)
        pose3d.y = float(point[2] / 100.0)
        pose3d.z = float(point[1] / 100.0)
        pose3d.h = h
        pose3d.q0 = quaternion[0]
        pose3d.q1 = quaternion[1]
        pose3d.q2 = quaternion[2]
        pose3d.q3 = quaternion[3]

        getbufferPose3d(obj_id, pose3d)

    def getSegment(self, current=None):
        rgblinelist = jderobot.bufferSegments()
        rgblinelist.buffer = []
        rgblinelist.refresh = refresh
        for i in bufferline[:]:
            rgblinelist.buffer.append(i)
            index = bufferline.index(i)
            del bufferline[index]
        return rgblinelist

    def getPoints(self, current=None):
        rgbpointlist = jderobot.bufferPoints()
        rgbpointlist.buffer = []
        rgbpointlist.refresh = refresh
        for i in bufferpoints[:]:
            rgbpointlist.buffer.append(i)
            index = bufferpoints.index(i)
            del bufferpoints[index]
        return rgbpointlist

    def getObj3D(self, id, current=None):
        if len(obj_list) > self.cont:
            obj3D = jderobot.object3d()
            model = obj_list[self.cont].split(":")
            if model[0] == "https":
                obj3D.obj = obj_list[self.cont]
                model = obj_list[self.cont].split("/")
                name,form = model[len(model)-1].split(".")
                print "Sending model by url: " + name + "." + form
            else:
                obj = open(obj_list[self.cont], "r").read()
                name,form = obj_list[self.cont].split(".")
                obj3D.obj = obj
                print "Sending model by file: " + name + "." + form
            pose3d = jderobot.Pose3DData()
            pose3d.x = float(init_pos[self.cont][0] / 100.)
            pose3d.y = float(init_pos[self.cont][1] / 100.)
            pose3d.z = float(init_pos[self.cont][2] / 100.)
            pose3d.h = 1
            pose3d.q0 = init_quat[self.cont][0]
            pose3d.q1 = init_quat[self.cont][1]
            pose3d.q2 = init_quat[self.cont][2]
            pose3d.q3 = init_quat[self.cont][3]
            id_list.append(id)
            obj3D.id = id
            obj3D.format = form
            obj3D.pos = pose3d
            obj3D.scale = scales[self.cont]
            self.cont = self.cont + 1
            return obj3D

    def getPoseObj3DData(self, current=None):
        pose3dlist = []
        for i in bufferpose3D[:]:
            pose3dlist.append(i)
            index = bufferpose3D.index(i)
            del bufferpose3D[index]
        return pose3dlist

    def clearAll(self, current=None):
        print "Clear All"


def getbufferSegment(seg, color, plane):
    rgbsegment = jderobot.RGBSegment()
    rgbsegment.seg = seg
    if not plane:
        rgbsegment.seg.fromPoint.z = rgbsegment.seg.fromPoint.z * uniform(1, 10)
        rgbsegment.seg.toPoint.z = rgbsegment.seg.toPoint.z * uniform(1, 10)
        rgbsegment.seg.fromPoint.y = rgbsegment.seg.fromPoint.y * uniform(1, 10)
        rgbsegment.seg.toPoint.y = rgbsegment.seg.toPoint.y * uniform(1, 10)
        rgbsegment.seg.fromPoint.x = rgbsegment.seg.fromPoint.x * uniform(1, 10)
        rgbsegment.seg.toPoint.x = rgbsegment.seg.toPoint.x * uniform(1, 10)
    rgbsegment.c = color
    bufferline.append(rgbsegment)


def getbufferPoint(point, color):
    rgbpoint = jderobot.RGBPoint()
    rgbpoint.x = point.x
    rgbpoint.y = point.y
    rgbpoint.z = point.z
    rgbpoint.r = color.r
    rgbpoint.g = color.g
    rgbpoint.b = color.b
    bufferpoints.append(rgbpoint)

def getbufferPose3d(id, pose):
    objpose3d = jderobot.PoseObj3D()
    objpose3d.id = id_list[id]
    objpose3d.pos = pose
    bufferpose3D.append(objpose3d)