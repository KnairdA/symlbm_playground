import numpy

from OpenGL.GL import *

class Cylinder:
    def __init__(self, x, y, z, r, h = 0, l = 0):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.h = h
        self.l = l

        self.circle = []
        for i in range(33):
            alpha = 2 * numpy.pi * (i/32)
            c0 = self.r * numpy.cos(alpha)
            c1 = self.r * numpy.sin(alpha)
            self.circle.append((c0, c1))

    def indicator(self):
        if self.h == 0:
            return lambda x, y, z: (x - self.x)**2 + (z - self.z)**2 < self.r*self.r and y >= self.y and y <= self.y+self.l
        else:
            return lambda x, y, z: (x - self.x)**2 + (y - self.y)**2 < self.r*self.r and z >= self.z and z <= self.z+self.h

    def draw(self):
        glBegin(GL_TRIANGLE_FAN)

        if self.h == 0:
            glNormal(0,-1, 0)
            glVertex(self.x, self.y, self.z)
            for (c0, c1) in self.circle:
                glNormal(0, -1, 0)
                glVertex(self.x+c0, self.y, self.z+c1)
        else:
            glNormal(0, 0, -1)
            glVertex(self.x, self.y, self.z)
            for (c0, c1) in self.circle:
                glVertex(self.x+c0, self.y+c1, self.z)

        glEnd()

        glBegin(GL_TRIANGLE_FAN)
        if self.h == 0:
            glNormal(0, 1, 0)
            glVertex(self.x, self.y+self.l, self.z)
            for (c0, c1) in self.circle:
                glVertex(self.x+c0, self.y+self.l, self.z+c1)
        else:
            glNormal(0, 0, 1)
            glVertex(self.x, self.y, self.z+self.h)
            for (c0, c1) in self.circle:
                glVertex(self.x+c0, self.y+c1, self.z+self.h)

        glEnd()

        glBegin(GL_TRIANGLE_STRIP)

        if self.h == 0:
            for (c0, c1) in self.circle:
                glNormal(c0, 0, c1)
                glVertex(self.x+c0, self.y, self.z+c1)
                glVertex(self.x+c0, self.y + self.l, self.z+c1)
        else:
            for (c0, c1) in self.circle:
                glNormal(c0, c1, 0)
                glVertex(self.x+c0, self.y+c1, self.z)
                glVertex(self.x+c0, self.y+c1, self.z+self.h)

        glEnd()

