import pyopencl as cl
mf = cl.mem_flags

import numpy
import time

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')

import sympy
import lbm_d2q9 as D2Q9

from mako.template import Template

class D2Q9_BGK_Lattice:
    def idx(self, x, y):
        return y * self.nX + x;

    def __init__(self, nX, nY, tau, geometry):
        self.nX = nX
        self.nY = nY
        self.nCells = nX * nY
        self.tau = tau
        self.tick = True

        self.platform = cl.get_platforms()[0]
        self.context  = cl.Context(properties=[(cl.context_properties.PLATFORM, self.platform)])
        self.queue = cl.CommandQueue(self.context)

        self.np_material = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.int32)
        self.setup_geometry(geometry)

        self.cl_pop_a = cl.Buffer(self.context, mf.READ_WRITE, size=9*self.nCells*numpy.float32(0).nbytes)
        self.cl_pop_b = cl.Buffer(self.context, mf.READ_WRITE, size=9*self.nCells*numpy.float32(0).nbytes)

        self.cl_moments  = cl.Buffer(self.context, mf.WRITE_ONLY, size=3*self.nCells*numpy.float32(0).nbytes)
        self.cl_material = cl.Buffer(self.context, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=self.np_material)

        self.build_kernel()

        self.program.equilibrilize(self.queue, (self.nX,self.nY), (32,1), self.cl_pop_a, self.cl_pop_b).wait()

    def setup_geometry(self, geometry):
        for y in range(1,self.nY-1):
            for x in range(1,self.nX-1):
                self.np_material[self.idx(x,y)] = geometry(self.nX,self.nY,x,y)

    def build_kernel(self):
        program_src = Template(filename = './template/kernel.mako').render(
            nX     = self.nX,
            nY     = self.nY,
            nCells = self.nCells,
            tau    = self.tau,
            moments_helper     = D2Q9.moments_opt[0],
            moments_assignment = D2Q9.moments_opt[1],
            collide_helper     = D2Q9.collide_opt[0],
            collide_assignment = D2Q9.collide_opt[1],
            c     = D2Q9.c,
            w     = D2Q9.w,
            ccode = sympy.ccode
        )
        self.program = cl.Program(self.context, program_src).build()

    def evolve(self):
        if self.tick:
            self.tick = False
            self.program.collide_and_stream(self.queue, (self.nX,self.nY), (32,1), self.cl_pop_a, self.cl_pop_b, self.cl_material)
        else:
            self.tick = True
            self.program.collide_and_stream(self.queue, (self.nX,self.nY), (32,1), self.cl_pop_b, self.cl_pop_a, self.cl_material)

    def sync(self):
        self.queue.finish()

    def get_moments(self):
        moments = numpy.ndarray(shape=(3, self.nCells), dtype=numpy.float32)
        if self.tick:
            self.program.collect_moments(self.queue, (self.nX,self.nY), (32,1), self.cl_pop_b, self.cl_moments)
        else:
            self.program.collect_moments(self.queue, (self.nX,self.nY), (32,1), self.cl_pop_a, self.cl_moments)
        cl.enqueue_copy(LBM.queue, moments, LBM.cl_moments).wait();
        return moments


def MLUPS(cells, steps, time):
    return cells * steps / time * 1e-6

def generate_moment_plots(lattice, moments):
    for i, m in enumerate(moments):
        print("Generating plot %d of %d." % (i+1, len(moments)))

        density = numpy.ndarray(shape=(lattice.nY-2, lattice.nX-2))
        for y in range(1,lattice.nY-1):
            for x in range(1,lattice.nX-1):
                density[y-1,x-1] = m[0,lattice.idx(x,y)]

        plt.figure(figsize=(10, 10))
        plt.imshow(density, origin='lower', vmin=0.2, vmax=2.0, cmap=plt.get_cmap('seismic'))
        plt.savefig("result/density_" + str(i) + ".png", bbox_inches='tight', pad_inches=0)

def box(nX, nY, x, y):
    if x == 1 or y == 1 or x == nX-2 or y == nY-2:
        return 2
    else:
        return 1

nUpdates = 1000
nStat    = 100

moments = []

print("Initializing simulation...\n")

LBM = D2Q9_BGK_Lattice(nX = 1024, nY = 1024, tau = 0.8, geometry = box)

print("Starting simulation using %d cells...\n" % LBM.nCells)

lastStat = time.time()

for i in range(1,nUpdates+1):
    LBM.evolve()

    if i % nStat == 0:
        LBM.sync()
        print("i = %4d; %3.0f MLUPS" % (i, MLUPS(LBM.nCells, nStat, time.time() - lastStat)))
        moments.append(LBM.get_moments())
        lastStat = time.time()

print("\nConcluded simulation.\n")

generate_moment_plots(LBM, moments)
