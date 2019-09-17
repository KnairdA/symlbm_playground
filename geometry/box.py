from OpenGL.GL import *

class Box:
    def __init__(self, x0, x1, y0, y1, z0, z1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1

    def indicator(self):
        return lambda x, y, z: x >= self.x0 and x <= self.x1 and y >= self.y0 and y <= self.y1 and z >= self.z0 and z <= self.z1

    def draw(self):
        glBegin(GL_POLYGON)
        glNormal(-1,0,0)
        glVertex(self.x0, self.y0, self.z0)
        glVertex(self.x0, self.y1, self.z0)
        glVertex(self.x0, self.y1, self.z1)
        glVertex(self.x0, self.y0, self.z1)
        glEnd()
        glBegin(GL_POLYGON)
        glNormal(1,0,0)
        glVertex(self.x1, self.y0, self.z0)
        glVertex(self.x1, self.y1, self.z0)
        glVertex(self.x1, self.y1, self.z1)
        glVertex(self.x1, self.y0, self.z1)
        glEnd()
        glBegin(GL_POLYGON)
        glNormal(0,0,1)
        glVertex(self.x0, self.y0, self.z1)
        glVertex(self.x1, self.y0, self.z1)
        glVertex(self.x1, self.y1, self.z1)
        glVertex(self.x0, self.y1, self.z1)
        glEnd()
        glBegin(GL_POLYGON)
        glNormal(0,0,-1)
        glVertex(self.x0, self.y0, self.z0)
        glVertex(self.x1, self.y0, self.z0)
        glVertex(self.x1, self.y1, self.z0)
        glVertex(self.x0, self.y1, self.z0)
        glEnd()
        glBegin(GL_POLYGON)
        glNormal(0,-1,0)
        glVertex(self.x0, self.y0, self.z0)
        glVertex(self.x1, self.y0, self.z0)
        glVertex(self.x1, self.y0, self.z1)
        glVertex(self.x0, self.y0, self.z1)
        glEnd()
        glBegin(GL_POLYGON)
        glNormal(0,1,0)
        glVertex(self.x0,self.y1,self.z0)
        glVertex(self.x1,self.y1,self.z0)
        glVertex(self.x1,self.y1,self.z1)
        glVertex(self.x0,self.y1,self.z1)
        glEnd()

