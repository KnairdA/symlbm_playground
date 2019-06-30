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


def pad(n, m):
    return (n // m + min(1,n % m)) * m

class Grid:
    def __init__(self, geometry, padding = None):
        if padding == None:
            self.size_x = geometry.size_x
            self.size_y = geometry.size_y
            self.size_z = geometry.size_z
        else:
            self.size_x = pad(geometry.size_x, padding[0])
            self.size_y = pad(geometry.size_y, padding[1])
            if geometry.size_z == 1:
                self.size_z = geometry.size_z
            else:
                self.size_z = pad(geometry.size_z, padding[2])

        self.volume = self.size_x * self.size_y * self.size_z

    def size(self):
        if self.size_z == 1:
            return (self.size_x, self.size_y)
        else:
            return (self.size_x, self.size_y, self.size_z)

class Memory:
    def __init__(self, descriptor, grid, context, float_type, align, opengl):
        self.descriptor = descriptor
        self.context    = context
        self.float_type = float_type

        if align:
            self.size_x = pad(grid.size_x, 32)
        else:
            self.size_x = grid.size_x

        self.size_y = grid.size_y
        self.size_z = grid.size_z

        self.volume = self.size_x * self.size_y * self.size_z

        self.pop_size     = descriptor.q     * self.volume * self.float_type(0).nbytes
        self.moments_size = (descriptor.d+1) * self.volume * self.float_type(0).nbytes

        self.cl_pop_a = cl.Buffer(self.context, mf.READ_WRITE, size=self.pop_size)
        self.cl_pop_b = cl.Buffer(self.context, mf.READ_WRITE, size=self.pop_size)

        if opengl:
            self.np_moments = numpy.ndarray(shape=(self.volume, 4), dtype=self.float_type)
            self.gl_moments = vbo.VBO(data=self.np_moments, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
            self.gl_moments.bind()
            self.cl_gl_moments  = cl.GLBuffer(self.context, mf.READ_WRITE, int(self.gl_moments))
        else:
            self.cl_moments  = cl.Buffer(self.context, mf.WRITE_ONLY, size=self.moments_size)

        self.cl_material = cl.Buffer(self.context, mf.READ_ONLY, size=self.volume * numpy.int32(0).nbytes)

    def gid(self, x, y, z = 0):
        return z * (self.size_x*self.size_y) + y * self.size_x + x;


class Lattice:
    def __init__(self,
        descriptor, geometry, moments, collide,
        pop_eq_src = '', boundary_src = '',
        platform = 0, precision = 'single', layout = None, padding = None, align = True, opengl = False
    ):
        self.descriptor = descriptor
        self.geometry   = geometry
        self.grid       = Grid(self.geometry, padding)

        self.float_type = {
            'single': (numpy.float32, 'float'),
            'double': (numpy.float64, 'double'),
        }.get(precision, None)

        self.platform = cl.get_platforms()[platform]

        if opengl:
            self.context = cl.Context(
                properties=[(cl.context_properties.PLATFORM, self.platform)] + get_gl_sharing_context_properties())
        else:
            self.context = cl.Context(
                properties=[(cl.context_properties.PLATFORM, self.platform)])

        self.queue = cl.CommandQueue(self.context)

        self.memory = Memory(self.descriptor, self.grid, self.context, self.float_type[0], align, opengl)
        self.tick = False

        self.moments = moments
        self.collide = collide

        self.pop_eq_src = pop_eq_src
        self.boundary_src = boundary_src

        self.layout = layout

        self.compiler_args = {
            'single': '-cl-single-precision-constant -cl-fast-relaxed-math',
            'double': '-cl-fast-relaxed-math'
        }.get(precision, None)

        self.build_kernel()

        self.program.equilibrilize(
            self.queue, self.grid.size(), self.layout, self.memory.cl_pop_a, self.memory.cl_pop_b).wait()

    def setup_geometry(self, material_at):
        material = numpy.ndarray(shape=(self.memory.volume, 1), dtype=numpy.int32)
        material[:,:] = 0

        for idx in self.geometry.inner_cells():
            material[self.memory.gid(*idx)] = material_at(self.geometry, *idx)

        cl.enqueue_copy(self.queue, self.memory.cl_material, material).wait();

    def build_kernel(self):
        program_src = Template(filename = str(Path(__file__).parent/'template/kernel.mako')).render(
            descriptor = self.descriptor,
            geometry   = self.geometry,
            memory     = self.memory,

            moments_subexpr    = self.moments[0],
            moments_assignment = self.moments[1],
            collide_subexpr    = self.collide[0],
            collide_assignment = self.collide[1],

            float_type = self.float_type[1],

            pop_eq_src = Template(self.pop_eq_src).render(
                descriptor = self.descriptor,
                geometry   = self.geometry,
                memory     = self.memory,
                float_type = self.float_type[1],
            ),
            boundary_src = Template(self.boundary_src).render(
                descriptor = self.descriptor,
                geometry   = self.geometry,
                memory     = self.memory,
                float_type = self.float_type[1],
            ),

            ccode = sympy.ccode
        )
        self.program = cl.Program(self.context, program_src).build(self.compiler_args)

    def evolve(self):
        if self.tick:
            self.tick = False
            self.program.collide_and_stream(
                self.queue, self.grid.size(), self.layout, self.memory.cl_pop_a, self.memory.cl_pop_b, self.memory.cl_material)
        else:
            self.tick = True
            self.program.collide_and_stream(
                self.queue, self.grid.size(), self.layout, self.memory.cl_pop_b, self.memory.cl_pop_a, self.memory.cl_material)

    def sync(self):
        self.queue.finish()

    def get_moments(self):
        moments = numpy.ndarray(shape=(self.descriptor.d+1, self.memory.volume), dtype=self.float_type[0])

        if self.tick:
            self.program.collect_moments(
                self.queue, self.grid.size(), self.layout, self.memory.cl_pop_b, self.memory.cl_moments)
        else:
            self.program.collect_moments(
                self.queue, self.grid.size(), self.layout, self.memory.cl_pop_a, self.memory.cl_moments)

        cl.enqueue_copy(self.queue, moments, self.memory.cl_moments).wait();

        return moments

    def sync_gl_moments(self):
        cl.enqueue_acquire_gl_objects(self.queue, [self.memory.cl_gl_moments])

        if self.tick:
            self.program.collect_gl_moments(
                self.queue, self.grid.size(), self.layout, self.memory.cl_pop_b, self.memory.cl_gl_moments)
        else:
            self.program.collect_gl_moments(
                self.queue, self.grid.size(), self.layout, self.memory.cl_pop_a, self.memory.cl_gl_moments)

        #cl.enqueue_release_gl_objects(self.queue, [self.cl_gl_moments])
        self.sync()
