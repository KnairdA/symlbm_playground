import pyopencl as cl
mf = cl.mem_flags

import numpy
import sympy

from mako.template import Template
from pathlib import Path

from OpenGL.GL import *


class MomentsTexture:
    def __init__(self, lattice, include_materials = True):
        self.lattice = lattice
        self.include_materials = include_materials
        self.gl_texture_buffer = numpy.ndarray(shape=(self.lattice.memory.volume, 4), dtype=self.lattice.memory.float_type)
        self.gl_texture_buffer[:,:] = 0.0

        self.gl_moments = glGenTextures(1)
        self.gl_texture_type = {2: GL_TEXTURE_2D, 3: GL_TEXTURE_3D}.get(self.lattice.descriptor.d)
        glBindTexture(self.gl_texture_type, self.gl_moments)

        if self.gl_texture_type == GL_TEXTURE_3D:
            glTexImage3D(self.gl_texture_type, 0, GL_RGBA32F, self.lattice.memory.size_x, self.lattice.memory.size_y, self.lattice.memory.size_z, 0, GL_RGBA, GL_FLOAT, self.gl_texture_buffer)
            glTexParameteri(self.gl_texture_type, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(self.gl_texture_type, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(self.gl_texture_type, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE)
            glTexParameteri(self.gl_texture_type, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE)
            glTexParameteri(self.gl_texture_type, GL_TEXTURE_WRAP_R,     GL_CLAMP_TO_EDGE)
            self.cl_gl_moments  = cl.GLTexture(self.lattice.context, mf.READ_WRITE, self.gl_texture_type, 0, self.gl_moments, 3)
        elif self.gl_texture_type == GL_TEXTURE_2D:
            glTexImage2D(self.gl_texture_type, 0, GL_RGBA32F, self.lattice.memory.size_x, self.lattice.memory.size_y, 0, GL_RGBA, GL_FLOAT, self.gl_texture_buffer)
            glTexParameteri(self.gl_texture_type, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(self.gl_texture_type, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(self.gl_texture_type, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE)
            glTexParameteri(self.gl_texture_type, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE)
            self.cl_gl_moments  = cl.GLTexture(self.lattice.context, mf.READ_WRITE, self.gl_texture_type, 0, self.gl_moments, 2)

        self.build_kernel()

    def build_kernel(self):
        program_src = Template(filename = str(Path(__file__).parent/'../template/opengl.mako')).render(
            descriptor = self.lattice.descriptor,
            geometry   = self.lattice.geometry,
            memory     = self.lattice.memory,

            moments_subexpr    = self.lattice.moments[0],
            moments_assignment = self.lattice.moments[1],
            collide_subexpr    = self.lattice.collide[0],
            collide_assignment = self.lattice.collide[1],

            float_type = self.lattice.float_type[1],

            ccode = sympy.ccode
        )
        self.program = cl.Program(self.lattice.context, program_src).build(self.lattice.compiler_args)

    def bind(self, location = GL_TEXTURE0):
        glEnable(self.gl_texture_type)
        glActiveTexture(location);
        glBindTexture(self.gl_texture_type, self.gl_moments)

    def collect_moments_from_pop_to_texture(self, population):
        if self.include_materials:
            self.program.collect_gl_moments_and_materials_to_texture(
                self.lattice.queue,
                self.lattice.grid.size(),
                self.lattice.layout,
                population,
                self.lattice.memory.cl_material,
                self.cl_gl_moments)
        else:
            self.program.collect_gl_moments_to_texture(
                self.lattice.queue,
                self.lattice.grid.size(),
                self.lattice.layout,
                population,
                self.cl_gl_moments)

    def collect(self):
        cl.enqueue_acquire_gl_objects(self.lattice.queue, [self.cl_gl_moments])

        if self.lattice.tick:
            self.collect_moments_from_pop_to_texture(self.lattice.memory.cl_pop_b)
        else:
            self.collect_moments_from_pop_to_texture(self.lattice.memory.cl_pop_a)
