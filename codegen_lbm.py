import pyopencl as cl
mf = cl.mem_flags

from string import Template

import numpy
import matplotlib.pyplot as plt

import time

kernel = """
unsigned int indexOfDirection(int i, int j) {
    return (i+1) + 3*(1-j);
}

unsigned int indexOfCell(int x, int y)
{
    return y * $nX + x;
}

unsigned int idx(int x, int y, int i, int j) {
    return indexOfDirection(i,j)*$nCells + indexOfCell(x,y);
}

__global float f_i(__global __read_only float* f, int x, int y, int i, int j) {
    return f[idx(x,y,i,j)];
}

__kernel void collide_and_stream(__global __write_only float* f_a,
                                 __global __read_only  float* f_b,
                                 __global __write_only float* moments,
                                 __global __read_only  int* material)
{
    const unsigned int gid = indexOfCell(get_global_id(0), get_global_id(1));

    const uint2 cell = (uint2)(get_global_id(0), get_global_id(1));

    const int m = material[gid];

    if ( m == 0 ) {
        return;
    }

    const float f_curr_0 = f_i(f_b, cell.x+1, cell.y-1, -1, 1);
    const float f_curr_1 = f_i(f_b, cell.x  , cell.y-1,  0, 1);
    const float f_curr_2 = f_i(f_b, cell.x-1, cell.y-1,  1, 1);
    const float f_curr_3 = f_i(f_b, cell.x+1, cell.y  , -1, 0);
    const float f_curr_4 = f_i(f_b, cell.x  , cell.y  ,  0, 0);
    const float f_curr_5 = f_i(f_b, cell.x-1, cell.y  ,  1, 0);
    const float f_curr_6 = f_i(f_b, cell.x+1, cell.y+1, -1,-1);
    const float f_curr_7 = f_i(f_b, cell.x  , cell.y+1,  0,-1);
    const float f_curr_8 = f_i(f_b, cell.x-1, cell.y+1,  1,-1);

    const float x0 = f_curr_0 + f_curr_1 + f_curr_2 + f_curr_3 + f_curr_4 + f_curr_5 + f_curr_6 + f_curr_7 + f_curr_8;
    const float x1 = 2*f_curr_0;
    const float x2 = 2*f_curr_8;
    const float x3 = -f_curr_3 + f_curr_5;
    const float x4 = pow(x0, -2);
    const float x5 = 9*x4;
    const float x6 = f_curr_0 - f_curr_8;
    const float x7 = f_curr_1 - f_curr_7;
    const float x8 = f_curr_2 - f_curr_6;
    const float x9 = x6 + x7 + x8;
    const float x10 = 6/x0;
    const float x11 = x10*x9;
    const float x12 = f_curr_3 - f_curr_5;
    const float x13 = -f_curr_2 + f_curr_6 + x12 + x6;
    const float x14 = pow(x13, 2);
    const float x15 = 3*x4;
    const float x16 = -x14*x15 + 2;
    const float x17 = x11 + x16;
    const float x18 = pow(x9, 2);
    const float x19 = x15*x18;
    const float x20 = -x19;
    const float x21 = x10*x13;
    const float x22 = x20 + x21;
    const float x23 = 1.0/$tau;
    const float x24 = (1.0/72.0)*x23;
    const float x25 = 6*x4;
    const float x26 = x18*x25;
    const float x27 = (1.0/18.0)*x23;
    const float x28 = x5*pow(2*f_curr_2 - 2*f_curr_6 + x3 + x7, 2);
    const float x29 = x20 - x21;
    const float x30 = x14*x25 + 2;
    const float x31 = -f_curr_0 + f_curr_8 + x3 + x8;
    const float x32 = x15*pow(x31, 2) - 2;
    const float x33 = x19 + x32;

    f_a[0*$nCells + gid] = f_curr_0 - x24*(72*f_curr_0 - x0*(x17 + x22 + x5*pow(-f_curr_1 + f_curr_7 - x1 + x2 + x3, 2)));
    f_a[1*$nCells + gid] = f_curr_1 - x27*(18*f_curr_1 - x0*(x17 + x26));
    f_a[2*$nCells + gid] = f_curr_2 - x24*(72*f_curr_2 - x0*(x17 + x28 + x29));
    f_a[3*$nCells + gid] = f_curr_3 - x27*(18*f_curr_3 - x0*(x22 + x30));
    f_a[4*$nCells + gid] = f_curr_4 - 1.0/9.0*x23*(9*f_curr_4 + 2*x0*x33);
    f_a[5*$nCells + gid] = f_curr_5 - x27*(18*f_curr_5 - x0*(x29 + x30));
    f_a[6*$nCells + gid] = f_curr_6 - x24*(72*f_curr_6 + x0*(x10*x31 + x11 - x28 + x33));
    f_a[7*$nCells + gid] = f_curr_7 - x27*(18*f_curr_7 + x0*(x11 - x26 + x32));
    f_a[8*$nCells + gid] = f_curr_8 - x24*(72*f_curr_8 - x0*(-x11 + x16 + x29 + x5*pow(x1 + x12 - x2 + x7, 2)));

    moments[gid] = x0;
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

        self.np_pop_a = numpy.ndarray(shape=(9, self.nCells), dtype=numpy.float32)
        self.np_pop_b = numpy.ndarray(shape=(9, self.nCells), dtype=numpy.float32)

        self.np_moments  = numpy.ndarray(shape=(3, self.nCells), dtype=numpy.float32)
        self.np_material = numpy.ndarray(shape=(self.nCells, 1), dtype=numpy.int32)

        self.setup_geometry()

        self.equilibrilize()
        self.setup_anomaly()

        self.cl_pop_a = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_a)
        self.cl_pop_b = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_pop_b)

        self.cl_material = cl.Buffer(self.context, mf.READ_ONLY  | mf.USE_HOST_PTR, hostbuf=self.np_material)
        self.cl_moments  = cl.Buffer(self.context, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=self.np_moments)

        self.build_kernel()

    def setup_geometry(self):
        self.np_material[:] = 0
        for x in range(1,self.nX-1):
            for y in range(1,self.nY-1):
                if x == 1 or y == 1 or x == self.nX-2 or y == self.nY-2:
                    self.np_material[self.idx(x,y)] = 2
                else:
                    self.np_material[self.idx(x,y)] = 1

    def equilibrilize(self):
        self.np_pop_a[(0,2,6,8),:] = 1./36.
        self.np_pop_a[(1,3,5,7),:] = 1./9.
        self.np_pop_a[4,:] = 4./9.

        self.np_pop_b[(0,2,6,8),:] = 1./36.
        self.np_pop_b[(1,3,5,7),:] = 1./9.
        self.np_pop_b[4,:] = 4./9.

    def setup_anomaly(self):
        bubbles = [ [        self.nX//4,        self.nY//4],
                    [        self.nX//4,self.nY-self.nY//4],
                    [self.nX-self.nX//4,        self.nY//4],
                    [self.nX-self.nX//4,self.nY-self.nY//4] ]

        for x in range(0,self.nX-1):
            for y in range(0,self.nY-1):
                for [a,b] in bubbles:
                    if numpy.sqrt((x-a)*(x-a)+(y-b)*(y-b)) < self.nX//10:
                        self.np_pop_a[:,self.idx(x,y)] = 1./24.
                        self.np_pop_b[:,self.idx(x,y)] = 1./24.

    def build_kernel(self):
        self.program = cl.Program(self.context, Template(kernel).substitute({
            'nX' : self.nX,
            'nY' : self.nY,
            'nCells': self.nCells,
            'tau': '0.8f'
        })).build() #'-cl-single-precision-constant -cl-fast-relaxed-math')

    def evolve(self):
        if self.tick:
            self.tick = False
            self.program.collide_and_stream(self.queue, (self.nX,self.nY), (64,1), self.cl_pop_a, self.cl_pop_b, self.cl_moments, self.cl_material)
        else:
            self.tick = True
            self.program.collide_and_stream(self.queue, (self.nX,self.nY), (64,1), self.cl_pop_b, self.cl_pop_a, self.cl_moments, self.cl_material)

    def sync(self):
        self.queue.finish()

    def show(self, i):
        cl.enqueue_copy(LBM.queue, LBM.np_moments, LBM.cl_moments).wait();

        density = numpy.ndarray(shape=(self.nX-2, self.nY-2))
        for y in range(1,self.nY-1):
            for x in range(1,self.nX-1):
                density[y-1,x-1] = self.np_moments[0,self.idx(x,y)]

        plt.imshow(density, vmin=0.2, vmax=2.0, cmap=plt.get_cmap("seismic"))
        plt.savefig("result/density_" + str(i) + ".png")


def MLUPS(cells, steps, time):
    return cells * steps / time * 1e-6

nUpdates = 1000
nStat = 100

print("Initializing simulation...\n")

LBM = D2Q9_BGK_Lattice(1024, 1024)

print("Starting simulation using %d cells...\n" % LBM.nCells)

lastStat = time.time()

for i in range(1,nUpdates+1):
    if i % nStat == 0:
        LBM.sync()
        #LBM.show(i)
        print("i = %4d; %3.0f MLUPS" % (i, MLUPS(LBM.nCells, nStat, time.time() - lastStat)))
        lastStat = time.time()

    LBM.evolve()

LBM.show(nUpdates)
