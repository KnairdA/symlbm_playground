import pyopencl as cl
mf = cl.mem_flags

import numpy
import sympy

from mako.template import Template
from pathlib import Path

from pyopencl.tools import get_gl_sharing_context_properties
import OpenGL.GL as gl
from OpenGL.arrays import vbo

class Geometry:
    def __init__(self, size_x, size_y, size_z = 1):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.volume = size_x * size_y * size_z

    def inner_cells(self):
        for idx in numpy.ndindex(self.inner_size()):
            yield tuple(map(lambda i: i + 1, idx))

    def size(self):
        if self.size_z == 1:
            return (self.size_x, self.size_y)
        else:
            return (self.size_x, self.size_y, self.size_z)

    def inner_size(self):
        if self.size_z == 1:
            return (self.size_x-2, self.size_y-2)
        else:
            return (self.size_x-2, self.size_y-2, self.size_z-2)

class Lattice:
    def __init__(self, descriptor, geometry, moments, collide, pop_eq_src = '', boundary_src = '', opengl = False):
        self.descriptor = descriptor
        self.geometry   = geometry

        self.moments = moments
        self.collide = collide

        self.pop_eq_src = pop_eq_src
        self.boundary_src = boundary_src

        self.platform = cl.get_platforms()[0]
        if opengl:
            self.context = cl.Context(
                properties=[(cl.context_properties.PLATFORM, self.platform)] + get_gl_sharing_context_properties())
        else:
            self.context = cl.Context(
                properties=[(cl.context_properties.PLATFORM, self.platform)])
        self.queue = cl.CommandQueue(self.context)

        self.np_material = numpy.ndarray(shape=(self.geometry.volume, 1), dtype=numpy.int32)

        self.tick = True

        self.pop_size     = descriptor.q     * self.geometry.volume * numpy.float32(0).nbytes
        self.moments_size = (descriptor.d+1) * self.geometry.volume * numpy.float32(0).nbytes

        self.cl_pop_a = cl.Buffer(self.context, mf.READ_WRITE, size=self.pop_size)
        self.cl_pop_b = cl.Buffer(self.context, mf.READ_WRITE, size=self.pop_size)

        if opengl:
            self.np_moments = numpy.ndarray(shape=(self.geometry.volume, 4), dtype=numpy.float32)
            self.gl_moments = vbo.VBO(data=self.np_moments, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
            self.gl_moments.bind()
            self.cl_gl_moments  = cl.GLBuffer(self.context, mf.READ_WRITE, int(self.gl_moments))
        else:
            self.cl_moments  = cl.Buffer(self.context, mf.WRITE_ONLY, size=self.moments_size)

        self.cl_material = cl.Buffer(self.context, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=self.np_material)

        self.build_kernel()

        self.layout = {
            (2, 9): (32,1),
            (3,19): (32,1,1),
            (3,27): (32,1,1)
        }.get((descriptor.d, descriptor.q), None)

        self.program.equilibrilize(
            self.queue, self.geometry.size(), self.layout, self.cl_pop_a, self.cl_pop_b).wait()

    def gid(self, x, y, z = 0):
        return z * (self.geometry.size_x*self.geometry.size_y) + y * self.geometry.size_x + x;

    def setup_geometry(self, material_at):
        for idx in self.geometry.inner_cells():
            self.np_material[self.gid(*idx)] = material_at(self.geometry, *idx)

        cl.enqueue_copy(self.queue, self.cl_material, self.np_material).wait();

    def build_kernel(self):
        program_src = Template(filename = str(Path(__file__).parent/'template/kernel.mako')).render(
            descriptor = self.descriptor,
            geometry   = self.geometry,

            moments_subexpr    = self.moments[0],
            moments_assignment = self.moments[1],
            collide_subexpr    = self.collide[0],
            collide_assignment = self.collide[1],

            pop_eq_src = Template(self.pop_eq_src).render(
                descriptor = self.descriptor,
                geometry   = self.geometry
            ),
            boundary_src = Template(self.boundary_src).render(
                descriptor = self.descriptor,
                geometry   = self.geometry
            ),

            ccode = sympy.ccode
        )
        self.program = cl.Program(self.context, program_src).build('-cl-single-precision-constant -cl-fast-relaxed-math')

    def evolve(self):
        if self.tick:
            self.tick = False
            self.program.collide_and_stream(
                self.queue, self.geometry.size(), self.layout, self.cl_pop_a, self.cl_pop_b, self.cl_material)
        else:
            self.tick = True
            self.program.collide_and_stream(
                self.queue, self.geometry.size(), self.layout, self.cl_pop_b, self.cl_pop_a, self.cl_material)

    def sync(self):
        self.queue.finish()

    def get_moments(self):
        moments = numpy.ndarray(shape=(self.descriptor.d+1, self.geometry.volume), dtype=numpy.float32)

        if self.tick:
            self.program.collect_moments(
                self.queue, self.geometry.size(), self.layout, self.cl_pop_b, self.cl_moments)
        else:
            self.program.collect_moments(
                self.queue, self.geometry.size(), self.layout, self.cl_pop_a, self.cl_moments)

        cl.enqueue_copy(self.queue, moments, self.cl_moments).wait();
        return moments

    def sync_gl_moments(self):
        cl.enqueue_acquire_gl_objects(self.queue, [self.cl_gl_moments])

        if self.tick:
            self.program.collect_gl_moments(
                self.queue, self.geometry.size(), self.layout, self.cl_pop_b, self.cl_gl_moments)
        else:
            self.program.collect_gl_moments(
                self.queue, self.geometry.size(), self.layout, self.cl_pop_a, self.cl_gl_moments)

        #cl.enqueue_release_gl_objects(self.queue, [self.cl_gl_moments])
        self.sync()
