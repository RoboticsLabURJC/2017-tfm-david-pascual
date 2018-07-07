import jderobot

import pointBuffer


class Viz3D:
    def __init__(self):
        self.MAXWORLD = 30
        color = (0, 0, 0)
        for i in range(0, self.MAXWORLD + 1):
            point1 = [-self.MAXWORLD * 1000 / 2 + i * 1000,
                      -self.MAXWORLD * 1000 / 2, 0]
            point2 = [-self.MAXWORLD * 1000 / 2 + i * 1000,
                      self.MAXWORLD * 1000 / 2, 0]
            point3 = [-self.MAXWORLD * 1000 / 2,
                      -self.MAXWORLD * 1000 / 2 + i * 1000, 0]
            point4 = [self.MAXWORLD * 1000 / 2,
                      -self.MAXWORLD * 1000 / 2 + i * 1000, 0]

            self.drawSegment(point1, point2, color)
            self.drawSegment(point3, point4, color)

    def drawPoint(self, point, color=(0, 0, 0)):
        pointJde = jderobot.Point()
        pointJde.x = float(point[0]) / 100.0
        pointJde.y = float(point[1] / 100.0)
        pointJde.z = float(point[2] / 100.0)

        colorJDE = jderobot.Color()
        colorJDE.r = float(color[0])
        colorJDE.g = float(color[1])
        colorJDE.b = float(color[2])
        pointBuffer.getbufferPoint(pointJde, colorJDE)
    
    def drawSegment(self, point_a, point_b, color=(0, 0, 0)):
        pointJde_a = jderobot.Point()
        pointJde_a.x = float(point_a[0]) / 100.0
        pointJde_a.y = float(point_a[1] / 100.0)
        pointJde_a.z = float(point_a[2] / 100.0)
        
        pointJde_b = jderobot.Point()
        pointJde_b.x = float(point_b[0]) / 100.0
        pointJde_b.y = float(point_b[1] / 100.0)
        pointJde_b.z = float(point_b[2] / 100.0)

        segJde = jderobot.Segment()
        segJde.fromPoint = pointJde_a
        segJde.toPoint = pointJde_b

        colorJDE = jderobot.Color()
        colorJDE.r = float(color[0])
        colorJDE.g = float(color[1])
        colorJDE.b = float(color[2])
        pointBuffer.getbufferSegment(segJde, colorJDE)
