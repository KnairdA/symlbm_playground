import pyopencl as cl
mf = cl.mem_flags

import numpy

from mako.template import Template
from pathlib import Path

import OpenGL.GL as gl
from OpenGL.arrays import vbo

class Particles:
    def __init__(self, lattice, grid):
        self.lattice = lattice
        self.context = self.lattice.context
        self.queue   = self.lattice.queue
        self.float_type = self.lattice.memory.float_type
        self.count = len(grid)

        self.np_particles = numpy.ndarray(shape=(self.count, 4), dtype=self.float_type)
        self.np_init_particles = numpy.ndarray(shape=(self.count, 4), dtype=self.float_type)

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

        self.cl_init_particles = cl.Buffer(self.context, mf.READ_ONLY, size=self.count * 4*numpy.float32(0).nbytes)
        cl.enqueue_copy(self.queue, self.cl_init_particles, self.np_init_particles).wait();

        self.build_kernel()

    def build_kernel(self):
        program_src = Template(filename = str(Path(__file__).parent/'../template/particles.mako')).render(
            descriptor = self.lattice.descriptor,
            geometry   = self.lattice.geometry,
            memory     = self.lattice.memory,
            float_type = self.float_type,
        )
        self.program = cl.Program(self.lattice.context, program_src).build(self.lattice.compiler_args)

    def bind(self):
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        self.gl_particles.bind()
        gl.glVertexPointer(4, gl.GL_FLOAT, 0, self.gl_particles)

    def update(self, aging = False):
        cl.enqueue_acquire_gl_objects(self.queue, [self.cl_gl_particles])

        if aging:
            age = numpy.float32(0.000006)
        else:
            age = numpy.float32(0.0)

        self.program.update_particles(
            self.queue, (self.count,1), None,
            self.lattice.memory.cl_moments,
            self.lattice.memory.cl_material,
            self.cl_gl_particles, self.cl_init_particles,
            age)
