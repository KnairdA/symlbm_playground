import pyopencl as cl
mf = cl.mem_flags

import numpy
import sympy

from mako.template import Template

class Geometry:
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y
        self.volume = size_x * size_y

    def inner_cells(self):
        for y in range(1,self.size_y-1):
            for x in range(1,self.size_x-1):
                yield x, y

    def span(self):
        return (self.size_x, self.size_y)

    def inner_span(self):
        return (self.size_x-2, self.size_y-2)


class Lattice:
    def __init__(self, descriptor, geometry, moments, collide, pop_eq_src = '', boundary_src = ''):
        self.descriptor = descriptor
        self.geometry   = geometry

        self.moments = moments
        self.collide = collide

        self.pop_eq_src = pop_eq_src
        self.boundary_src = boundary_src

        self.platform = cl.get_platforms()[0]
        self.context  = cl.Context(properties=[(cl.context_properties.PLATFORM, self.platform)])
        self.queue = cl.CommandQueue(self.context)

        self.np_material = numpy.ndarray(shape=(self.geometry.volume, 1), dtype=numpy.int32)

        self.tick = True

        self.pop_size     = descriptor.q     * self.geometry.volume * numpy.float32(0).nbytes
        self.moments_size = (descriptor.d+1) * self.geometry.volume * numpy.float32(0).nbytes

        self.cl_pop_a = cl.Buffer(self.context, mf.READ_WRITE, size=self.pop_size)
        self.cl_pop_b = cl.Buffer(self.context, mf.READ_WRITE, size=self.pop_size)

        self.cl_moments  = cl.Buffer(self.context, mf.WRITE_ONLY, size=self.moments_size)
        self.cl_material = cl.Buffer(self.context, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=self.np_material)

        self.build_kernel()

        self.program.equilibrilize(
            self.queue, (self.geometry.size_x,self.geometry.size_y), (32,1), self.cl_pop_a, self.cl_pop_b).wait()

    def idx(self, x, y):
        return y * self.geometry.size_x + x;

    def setup_geometry(self, material_at):
        for x, y in self.geometry.inner_cells():
            self.np_material[self.idx(x,y)] = material_at(self.geometry, x, y)

        cl.enqueue_copy(self.queue, self.cl_material, self.np_material).wait();

    def build_kernel(self):
        program_src = Template(filename = './template/kernel.mako').render(
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
                self.queue, self.geometry.span(), (32,1), self.cl_pop_a, self.cl_pop_b, self.cl_material)
        else:
            self.tick = True
            self.program.collide_and_stream(
                self.queue, self.geometry.span(), (32,1), self.cl_pop_b, self.cl_pop_a, self.cl_material)

    def sync(self):
        self.queue.finish()

    def get_moments(self):
        moments = numpy.ndarray(shape=(self.descriptor.d+1, self.geometry.volume), dtype=numpy.float32)
        if self.tick:
            self.program.collect_moments(
                self.queue, self.geometry.span(), (32,1), self.cl_pop_b, self.cl_moments)
        else:
            self.program.collect_moments(
                self.queue, self.geometry.span(), (32,1), self.cl_pop_a, self.cl_moments)
        cl.enqueue_copy(self.queue, moments, self.cl_moments).wait();
        return moments
