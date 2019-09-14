import pyopencl as cl
mf = cl.mem_flags

import numpy

import OpenGL.GL as gl
from OpenGL.arrays import vbo

class Particles:
    def __init__(self, context, queue, float_type, grid):
        self.context = context
        self.count = len(grid)

        self.np_particles = numpy.ndarray(shape=(self.count, 4), dtype=float_type)
        self.np_init_particles = numpy.ndarray(shape=(self.count, 4), dtype=float_type)

        if len(grid[0,:]) == 2:
            self.np_particles[:,0:2] = grid
            self.np_particles[:,2] = 0
            self.np_particles[:,3] = numpy.random.sample(self.count)
            self.np_init_particles = self.np_particles
        elif len(grid[0,:]) == 3:
            self.np_particles[:,0:3] = grid
            self.np_particles[:,3]   = numpy.random.sample(self.count)
            self.np_init_particles = self.np_particles

        self.gl_particles = vbo.VBO(data=self.np_particles, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
        self.gl_particles.bind()
        self.cl_gl_particles = cl.GLBuffer(self.context, mf.READ_WRITE, int(self.gl_particles))

        self.cl_init_particles = cl.Buffer(context, mf.READ_ONLY, size=self.count * 4*numpy.float32(0).nbytes)
        cl.enqueue_copy(queue, self.cl_init_particles, self.np_init_particles).wait();
