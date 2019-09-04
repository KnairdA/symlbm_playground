import pyopencl as cl
mf = cl.mem_flags

import numpy

import OpenGL.GL as gl
from OpenGL.arrays import vbo

class Particles:
    def __init__(self, context, float_type, grid):
        self.context = context
        self.count = len(grid)

        self.np_particles = numpy.ndarray(shape=(self.count, 4), dtype=float_type)

        self.np_particles[:,0:2] = grid
        self.np_particles[:,2:4] = self.np_particles[:,0:2]

        self.gl_particles = vbo.VBO(data=self.np_particles, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
        self.gl_particles.bind()
        self.cl_gl_particles = cl.GLBuffer(self.context, mf.READ_WRITE, int(self.gl_particles))

