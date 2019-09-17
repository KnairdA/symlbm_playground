import numpy

from OpenGL.GL import *

class Sphere:
    def __init__(self, x, y, z, r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r

    def indicator(self):
        return lambda x, y, z: (x - self.x)**2 + (y - self.y)**2 + (z - self.z)**2 < self.r*self.r

    def draw(self, resolution = 32):
        for i in range(0,resolution+1):
            lat0 = numpy.pi * (-0.5 + (i - 1) / resolution)
            z0   = numpy.sin(lat0)
            zr0  = numpy.cos(lat0)

            lat1 = numpy.pi * (-0.5 + i / resolution)
            z1   = numpy.sin(lat1)
            zr1  = numpy.cos(lat1)

            glBegin(GL_QUAD_STRIP)
            for j in range(0,resolution+1):
                lng = 2 * numpy.pi * (j - 1) / resolution
                x = numpy.cos(lng)
                y = numpy.sin(lng)

                glNormal(x * zr0, y * zr0, z0)
                glVertex(self.x + self.r * x * zr0, self.y + self.r * y * zr0, self.z + self.r * z0)
                glNormal(x * zr1, y * zr1, z1)
                glVertex(self.x + self.r * x * zr1, self.y + self.r * y * zr1, self.z + self.r * z1)

            glEnd()
