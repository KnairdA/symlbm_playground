import pyopencl as cl
mf = cl.mem_flags

import numpy

from mako.template import Template
from pathlib import Path

from OpenGL.GL import *
from OpenGL.arrays import vbo

class Streamlines:
    def __init__(self, lattice, moments, origins):
        self.lattice = lattice
        self.context = self.lattice.context
        self.queue   = self.lattice.queue
        self.float_type = self.lattice.memory.float_type
        self.moments = moments

        self.count = len(origins)
        self.np_origins = numpy.ndarray(shape=(self.count, 2), dtype=self.float_type)

        for i, pos in enumerate(origins):
            self.np_origins[i,:] = pos

        self.cl_origins = cl.Buffer(self.context, mf.READ_ONLY, size=self.count * 2*numpy.float32(0).nbytes)
        cl.enqueue_copy(self.queue, self.cl_origins, self.np_origins).wait();

        self.gl_texture_buffer = numpy.ndarray(shape=(self.lattice.memory.volume, 4), dtype=self.lattice.memory.float_type)
        self.gl_texture_buffer[:,:] = 0.0

        self.gl_streamlines = glGenTextures(1)
        self.gl_texture_type = GL_TEXTURE_2D
        glBindTexture(self.gl_texture_type, self.gl_streamlines)

        glTexImage2D(self.gl_texture_type, 0, GL_RGBA32F, self.lattice.memory.size_x, self.lattice.memory.size_y, 0, GL_RGBA, GL_FLOAT, self.gl_texture_buffer)
        glTexParameteri(self.gl_texture_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(self.gl_texture_type, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(self.gl_texture_type, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE)
        glTexParameteri(self.gl_texture_type, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE)
        self.cl_gl_streamlines = cl.GLTexture(self.lattice.context, mf.READ_WRITE, self.gl_texture_type, 0, self.gl_streamlines, 2)

        self.build_kernel()

    def build_kernel(self):
        program_src = Template(filename = str(Path(__file__).parent/'../template/streamline.mako')).render(
            descriptor = self.lattice.descriptor,
            geometry   = self.lattice.geometry,
            memory     = self.lattice.memory,
            float_type = self.float_type,
        )
        self.program = cl.Program(self.lattice.context, program_src).build(self.lattice.compiler_args)

    def bind(self, location = GL_TEXTURE0):
        glEnable(self.gl_texture_type)
        glActiveTexture(location);
        glBindTexture(self.gl_texture_type, self.gl_streamlines)

    def update(self):
        cl.enqueue_acquire_gl_objects(self.queue, [self.cl_gl_streamlines])

        self.program.dillute(
            self.queue, (self.lattice.memory.size_x,self.lattice.memory.size_y), None,
            self.lattice.memory.cl_material,
            self.cl_gl_streamlines)

        self.program.draw_streamline(
            self.queue, (self.count,1), None,
            self.moments.cl_gl_moments,
            self.lattice.memory.cl_material,
            self.cl_origins,
            self.cl_gl_streamlines)
