import pyopencl as cl
mf = cl.mem_flags
from pyopencl.tools import get_gl_sharing_context_properties

from string import Template

import numpy
import matplotlib.pyplot as plt

from timeit import default_timer as timer

kernel = """
float constant w[9] = {
    1./36., 1./9., 1./36.,
    1./9. , 4./9., 1./9. ,
    1./36 , 1./9., 1./36.
};

uint2 cellAtGid(unsigned int gid)
{
    const int y = gid / $nX;
    return (uint2)(gid - $nX*y, y);
}

unsigned int gidOfCell(int x, int y)
{
    return y * $nX + x;
}

unsigned int indexOfDirection(int i, int j) {
    return 3*(i+1) + (j+1);
}

float comp(int i, int j, float2 v) {
    return i*v.x + j*v.y;
}

float sq(float x) {
    return x*x;
}

float equilibrium(float d, float2 v, int i, int j) {
    return w[indexOfDirection(i,j)] * d * (1 + 3*comp(i,j,v) + 4.5*sq(comp(i,j,v)) - 1.5*dot(v,v));
}

float bgk(__global const float* pop, uint ngid, int i, int j, float d, float2 v) {
    return pop[ngid] + $tau * (equilibrium(d,v,i,j) - pop[ngid]);
}

__kernel void collide_and_stream(__global float* pop_a_0,
                                 __global float* pop_a_1,
                                 __global float* pop_a_2,
                                 __global float* pop_a_3,
                                 __global float* pop_a_4,
                                 __global float* pop_a_5,
                                 __global float* pop_a_6,
                                 __global float* pop_a_7,
                                 __global float* pop_a_8,
                                 __global const float* pop_b_0,
                                 __global const float* pop_b_1,
                                 __global const float* pop_b_2,
                                 __global const float* pop_b_3,
                                 __global const float* pop_b_4,
                                 __global const float* pop_b_5,
                                 __global const float* pop_b_6,
                                 __global const float* pop_b_7,
                                 __global const float* pop_b_8,
                                 __global float* moments,
                                 __global const int* material)
{
    const unsigned int gid = get_global_id(0);
    const uint2 cell = cellAtGid(gid);

    const int m = material[gid];

    if ( m == 0 ) {
        return;
    }

    const float  d = pop_b_0[gid] + pop_b_1[gid] + pop_b_2[gid] + pop_b_3[gid] + pop_b_4[gid] + pop_b_5[gid] + pop_b_6[gid] + pop_b_7[gid] + pop_b_8[gid];

    const float2 v = (float2)(
        (pop_b_5[gid] - pop_b_3[gid] + pop_b_2[gid] - pop_b_6[gid] + pop_b_8[gid] - pop_b_0[gid]) / d,
        (pop_b_1[gid] - pop_b_7[gid] + pop_b_2[gid] - pop_b_6[gid] - pop_b_8[gid] + pop_b_0[gid]) / d
    );

    if ( m == 1 ) {
        pop_a_0[gid] = bgk(pop_b_0, gidOfCell(cell.x+1, cell.y-1), -1, 1, d, v);
        pop_a_1[gid] = bgk(pop_b_1, gidOfCell(cell.x  , cell.y-1),  0, 1, d, v);
        pop_a_2[gid] = bgk(pop_b_2, gidOfCell(cell.x-1, cell.y-1),  1, 1, d, v);

        pop_a_3[gid] = bgk(pop_b_3, gidOfCell(cell.x+1, cell.y  ), -1, 0, d, v);
        pop_a_4[gid] = bgk(pop_b_4, gidOfCell(cell.x  , cell.y  ),  0, 0, d, v);
        pop_a_5[gid] = bgk(pop_b_5, gidOfCell(cell.x-1, cell.y  ),  1, 0, d, v);

        pop_a_6[gid] = bgk(pop_b_6, gidOfCell(cell.x+1, cell.y+1), -1,-1, d, v);
        pop_a_7[gid] = bgk(pop_b_7, gidOfCell(cell.x  , cell.y+1),  0,-1, d, v);
        pop_a_8[gid] = bgk(pop_b_8, gidOfCell(cell.x-1, cell.y+1),  1,-1, d, v);
    } else {
        pop_a_8[gid] = bgk(pop_b_0, gidOfCell(cell.x+1, cell.y-1), -1, 1, d, v);
        pop_a_7[gid] = bgk(pop_b_1, gidOfCell(cell.x  , cell.y-1),  0, 1, d, v);
        pop_a_6[gid] = bgk(pop_b_2, gidOfCell(cell.x-1, cell.y-1),  1, 1, d, v);

        pop_a_5[gid] = bgk(pop_b_3, gidOfCell(cell.x+1, cell.y  ), -1, 0, d, v);
        pop_a_4[gid] = bgk(pop_b_4, gidOfCell(cell.x  , cell.y  ),  0, 0, d, v);
        pop_a_3[gid] = bgk(pop_b_5, gidOfCell(cell.x-1, cell.y  ),  1, 0, d, v);

        pop_a_2[gid] = bgk(pop_b_6, gidOfCell(cell.x+1, cell.y+1), -1,-1, d, v);
        pop_a_1[gid] = bgk(pop_b_7, gidOfCell(cell.x  , cell.y+1),  0,-1, d, v);
        pop_a_0[gid] = bgk(pop_b_8, gidOfCell(cell.x-1, cell.y+1),  1,-1, d, v);
    }

    moments[gid*3+0] = d;
    moments[gid*3+1] = v.x;
    moments[gid*3+2] = v.y;
}"""

class D2Q9_BGK_Lattice:
    def idx(self, x, y):
        return y * self.nX + x;

    def __init__(self, nX, nY):
        self.nX = nX
        self.nY = nY
        self.nCells = nX * nY
        self.tick = True

        self.platform = cl.get_platforms()[0]
        self.context  = cl.Context(properties=[(cl.context_properties.PLATFORM, self.platform)])
        self.queue = cl.CommandQueue(self.context)

        self.np_pop_a_0 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_a_1 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_a_2 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_a_3 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_a_4 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_a_5 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_a_6 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_a_7 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_a_8 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)

        self.np_pop_b_0 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_b_1 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_b_2 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_b_3 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_b_4 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_b_5 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_b_6 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_b_7 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)
        self.np_pop_b_8 = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.float32)

        self.np_moments  = numpy.ndarray(shape=(self.nCells, 3), dtype=numpy.float32)
        self.np_material = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.int32)

        self.setup_geometry()

        self.equilibrilize()
        self.setup_anomaly()

        self.cl_pop_a_0 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_a_0)
        self.cl_pop_a_1 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_a_1)
        self.cl_pop_a_2 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_a_2)
        self.cl_pop_a_3 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_a_3)
        self.cl_pop_a_4 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_a_4)
        self.cl_pop_a_5 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_a_5)
        self.cl_pop_a_6 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_a_6)
        self.cl_pop_a_7 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_a_7)
        self.cl_pop_a_8 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_a_8)

        self.cl_pop_b_0 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_b_0)
        self.cl_pop_b_1 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_b_1)
        self.cl_pop_b_2 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_b_2)
        self.cl_pop_b_3 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_b_3)
        self.cl_pop_b_4 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_b_4)
        self.cl_pop_b_5 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_b_5)
        self.cl_pop_b_6 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_b_6)
        self.cl_pop_b_7 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_b_7)
        self.cl_pop_b_8 = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_b_8)

        self.cl_moments  = cl.Buffer(self.context, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=self.np_moments)
        self.cl_material = cl.Buffer(self.context, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=self.np_material)

        self.build_kernel()

    def setup_geometry(self):
        self.np_material[:] = 0
        for x in range(1,self.nX-1):
            for y in range(1,self.nY-1):
                if x == 1 or y == 1 or x == self.nX-2 or y == self.nY-2:
                    self.np_material[self.idx(x,y)] = -1
                else:
                    self.np_material[self.idx(x,y)] = 1

    def equilibrilize(self):
        self.np_pop_a_0[:] = 1./36.
        self.np_pop_a_1[:] = 1./9.
        self.np_pop_a_2[:] = 1./36.
        self.np_pop_a_3[:] = 1./9.
        self.np_pop_a_4[:] = 4./9.
        self.np_pop_a_5[:] = 1./9.
        self.np_pop_a_6[:] = 1./36
        self.np_pop_a_7[:] = 1./9.
        self.np_pop_a_8[:] = 1./36.

        self.np_pop_b_0[:] = 1./36.
        self.np_pop_b_1[:] = 1./9.
        self.np_pop_b_2[:] = 1./36.
        self.np_pop_b_3[:] = 1./9.
        self.np_pop_b_4[:] = 4./9.
        self.np_pop_b_5[:] = 1./9.
        self.np_pop_b_6[:] = 1./36
        self.np_pop_b_7[:] = 1./9.
        self.np_pop_b_8[:] = 1./36.


    def setup_anomaly(self):
        bubbles = [ [        self.nX//4,        self.nY//4],
                    [        self.nX//4,self.nY-self.nY//4],
                    [self.nX-self.nX//4,        self.nY//4],
                    [self.nX-self.nX//4,self.nY-self.nY//4] ]

        for x in range(0,self.nX-1):
            for y in range(0,self.nY-1):
                for [a,b] in bubbles:
                    if numpy.sqrt((x-a)*(x-a)+(y-b)*(y-b)) < self.nX//10:
                        self.np_pop_a_0[self.idx(x,y)] = 1./24.
                        self.np_pop_a_1[self.idx(x,y)] = 1./24.
                        self.np_pop_a_2[self.idx(x,y)] = 1./24.
                        self.np_pop_a_3[self.idx(x,y)] = 1./24.
                        self.np_pop_a_4[self.idx(x,y)] = 1./24.
                        self.np_pop_a_5[self.idx(x,y)] = 1./24.
                        self.np_pop_a_6[self.idx(x,y)] = 1./24.
                        self.np_pop_a_7[self.idx(x,y)] = 1./24.
                        self.np_pop_a_8[self.idx(x,y)] = 1./24.

                        self.np_pop_b_0[self.idx(x,y)] = 1./24.
                        self.np_pop_b_1[self.idx(x,y)] = 1./24.
                        self.np_pop_b_2[self.idx(x,y)] = 1./24.
                        self.np_pop_b_3[self.idx(x,y)] = 1./24.
                        self.np_pop_b_4[self.idx(x,y)] = 1./24.
                        self.np_pop_b_5[self.idx(x,y)] = 1./24.
                        self.np_pop_b_6[self.idx(x,y)] = 1./24.
                        self.np_pop_b_7[self.idx(x,y)] = 1./24.
                        self.np_pop_b_8[self.idx(x,y)] = 1./24.

    def build_kernel(self):
        self.program = cl.Program(self.context, Template(kernel).substitute({
            'nX' : self.nX,
            'nY' : self.nY,
            'tau': 0.56
        })).build()

    def evolve(self):
        if self.tick:
            self.tick = False
            self.program.collide_and_stream(self.queue, (self.nCells,), None,
                self.cl_pop_a_0,
                self.cl_pop_a_1,
                self.cl_pop_a_2,
                self.cl_pop_a_3,
                self.cl_pop_a_4,
                self.cl_pop_a_5,
                self.cl_pop_a_6,
                self.cl_pop_a_7,
                self.cl_pop_a_8,
                self.cl_pop_b_0,
                self.cl_pop_b_1,
                self.cl_pop_b_2,
                self.cl_pop_b_3,
                self.cl_pop_b_4,
                self.cl_pop_b_5,
                self.cl_pop_b_6,
                self.cl_pop_b_7,
                self.cl_pop_b_8,
                self.cl_moments,
                self.cl_material)
            self.queue.finish()
        else:
            self.tick = True
            self.program.collide_and_stream(self.queue, (self.nCells,), None,
                self.cl_pop_b_0,
                self.cl_pop_b_1,
                self.cl_pop_b_2,
                self.cl_pop_b_3,
                self.cl_pop_b_4,
                self.cl_pop_b_5,
                self.cl_pop_b_6,
                self.cl_pop_b_7,
                self.cl_pop_b_8,
                self.cl_pop_a_0,
                self.cl_pop_a_1,
                self.cl_pop_a_2,
                self.cl_pop_a_3,
                self.cl_pop_a_4,
                self.cl_pop_a_5,
                self.cl_pop_a_6,
                self.cl_pop_a_7,
                self.cl_pop_a_8,
                self.cl_moments,
                self.cl_material)
            self.queue.finish()

    def show(self, i):
        cl.enqueue_copy(self.queue, self.np_moments, self.cl_moments).wait();

        density = numpy.ndarray(shape=(self.nX, self.nY))

        for y in range(0,self.nY-1):
            for x in range(0,self.nX-1):
                density[x,y] = self.np_moments[self.idx(x,y),0]

        plt.imshow(density, vmin=0.2, vmax=2, cmap=plt.get_cmap("seismic"))
        plt.savefig("result/density_" + str(i) + ".png")


def MLUPS(cells, steps, time):
    return ((cells*steps) / time) / 1000000

LBM = D2Q9_BGK_Lattice(1000, 1000)

nUpdates = 10000

start = timer()

for i in range(1,nUpdates):
    LBM.evolve()

end = timer()

runtime = end - start

print("Cells:   " + str(LBM.nCells))
print("Updates: " + str(nUpdates))
print("Time:    " + str(runtime))
print("MLUPS:   " + str(MLUPS(LBM.nCells, nUpdates, end - start)))
