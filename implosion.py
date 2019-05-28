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

float pop(__global const float* cell, int i, int j) {
    return cell[indexOfDirection(i,j)];
}

float comp(int i, int j, float2 v) {
    return i*v.x + j*v.y;
}

float sq(float x) {
    return x*x;
}

float norm(float2 v) {
    return sqrt(dot(v,v));
}

float density(__global const float* cell) {
    float d = 0.;
    for ( int i = 0; i < 9; ++i ) {
        d += cell[i];
    }
    return d;
}

float2 velocity(__global const float* cell, float d)
{
    return (float2)(
        (pop(cell,1,0) - pop(cell,-1,0) + pop(cell,1,1) - pop(cell,-1,-1) + pop(cell,1,-1) - pop(cell,-1,1)) / d,
        (pop(cell,0,1) - pop(cell,0,-1) + pop(cell,1,1) - pop(cell,-1,-1) - pop(cell,1,-1) + pop(cell,-1,1)) / d
    );
}

float equilibrium(float d, float2 v, int i, int j) {
    return w[indexOfDirection(i,j)] * d * (1 + 3*comp(i,j,v) + 4.5*sq(comp(i,j,v)) - 1.5*sq(norm(v)));
}

__kernel void collide_and_stream(__global float* pop_a,
                                 __global const float* pop_b,
                                 __global const int* material)
{
    const unsigned int gid = get_global_id(0);
    const uint2 cell = cellAtGid(gid);

    const int m = material[gid];

    if ( m == 0 ) {
        return;
    }

    const float  d = density(&pop_b[gid*9]);
    const float2 v = velocity(&pop_b[gid*9],d);

    for ( int i = -1; i <= 1; ++i ) {
        for ( int j = -1; j <= 1; ++j ) {
            const unsigned int ngid = gidOfCell(cell.x-i, cell.y-j);
            pop_a[gid*9 + indexOfDirection(m*i,m*j)] =
                pop_b[ngid*9 + indexOfDirection(i,j)] + $tau * (equilibrium(d,v,i,j) - pop_b[ngid*9 + indexOfDirection(i,j)]);
        }
    }
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

        self.np_pop_a = numpy.ndarray(shape=(self.nCells, 9), dtype=numpy.float32)
        self.np_pop_b = numpy.ndarray(shape=(self.nCells, 9), dtype=numpy.float32)
        self.np_material = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.int32)

        self.setup_geometry()

        self.equilibrilize()
        self.setup_anomaly()

        self.cl_pop_a = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_a)
        self.cl_pop_b = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_b)
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
        self.np_pop_a[:,:] = [ 1./36., 1./9., 1./36., 1./9. , 4./9., 1./9. , 1./36 , 1./9., 1./36. ]
        self.np_pop_b[:,:] = [ 1./36., 1./9., 1./36., 1./9. , 4./9., 1./9. , 1./36 , 1./9., 1./36. ]

    def setup_anomaly(self):
        bubbles = [ [        self.nX//4,        self.nY//4],
                    [        self.nX//4,self.nY-self.nY//4],
                    [self.nX-self.nX//4,        self.nY//4],
                    [self.nX-self.nX//4,self.nY-self.nY//4] ]

        for x in range(0,self.nX-1):
            for y in range(0,self.nY-1):
                for [a,b] in bubbles:
                    if numpy.sqrt((x-a)*(x-a)+(y-b)*(y-b)) < self.nX//10:
                        self.np_pop_a[self.idx(x,y),:] = 1./24.
                        self.np_pop_b[self.idx(x,y),:] = 1./24.

    def build_kernel(self):
        self.program = cl.Program(self.context, Template(kernel).substitute({
            'nX' : self.nX,
            'nY' : self.nY,
            'tau': 0.56
        })).build()

    def evolve(self):
        if self.tick:
            self.tick = False
            self.program.collide_and_stream(self.queue, (self.nCells,), None, self.cl_pop_a, self.cl_pop_b, self.cl_material)
            self.queue.finish()
        else:
            self.tick = True
            self.program.collide_and_stream(self.queue, (self.nCells,), None, self.cl_pop_b, self.cl_pop_a, self.cl_material)
            self.queue.finish()

    def show(self, i):
        if self.tick:
            cl.enqueue_copy(LBM.queue, LBM.np_pop_a, LBM.cl_pop_b).wait();
        else:
            cl.enqueue_copy(LBM.queue, LBM.np_pop_a, LBM.cl_pop_a).wait();

        pop = numpy.ndarray(shape=(self.nX, self.nY))

        for y in range(0,self.nY-1):
            for x in range(0,self.nX-1):
                pop[x,y] = numpy.sum(self.np_pop_a[self.idx(x,y),:])

        plt.imshow(pop, vmin=0.2, vmax=2, cmap=plt.get_cmap("seismic"))
        plt.savefig("result/density_" + str(i) + ".png")


def MLUPS(cells, steps, time):
    return ((cells*steps) / time) / 1000000

LBM = D2Q9_BGK_Lattice(1024, 1024)

nUpdates = 100

start = timer()

for i in range(0,nUpdates):
    LBM.evolve()

end = timer()

runtime = end - start

print("Cells:   " + str(LBM.nCells))
print("Updates: " + str(nUpdates))
print("Time:    " + str(runtime))
print("MLUPS:   " + str(MLUPS(LBM.nCells, nUpdates, end - start)))
