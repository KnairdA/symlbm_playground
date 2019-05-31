import pyopencl as cl
mf = cl.mem_flags

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

unsigned int indexOfDirection(int i, int j) {
    return (i+1) + 3*(1-j);
}

unsigned int indexOfCell(int x, int y)
{
    return y * $nX + x;
}

unsigned int idx(int x, int y, int i, int j) {
    return indexOfDirection(i,j)*$nX*$nY + indexOfCell(x,y);
}

uint2 cellAtIndex(unsigned int gid)
{
    const int y = gid / $nX;
    return (uint2)(gid - $nX*y, y);
}


__global float* f_i(__global float* f, int x, int y, int i, int j) {
    return f + idx(x,y,i,j);
}

float comp(int i, int j, float2 v) {
    return i*v.x + j*v.y;
}

float sq(float x) {
    return x*x;
}

float density(__global const float* f, unsigned int gid) {
    return f[0*$nX*$nY + gid]
         + f[1*$nX*$nY + gid]
         + f[2*$nX*$nY + gid]
         + f[3*$nX*$nY + gid]
         + f[4*$nX*$nY + gid]
         + f[5*$nX*$nY + gid]
         + f[6*$nX*$nY + gid]
         + f[7*$nX*$nY + gid]
         + f[8*$nX*$nY + gid];
}

float2 velocity(__global const float* f, float d, unsigned int gid)
{
    return (float2)(
        (f[5*$nX*$nY+gid] - f[3*$nX*$nY+gid] + f[2*$nX*$nY+gid] - f[6*$nX*$nY+gid] + f[8*$nX*$nY+gid] - f[0*$nX*$nY+gid]) / d,
        (f[1*$nX*$nY+gid] - f[7*$nX*$nY+gid] + f[2*$nX*$nY+gid] - f[6*$nX*$nY+gid] - f[8*$nX*$nY+gid] + f[0*$nX*$nY+gid]) / d
    );
}

float f_eq(float d, float2 v, int i, int j) {
    return w[indexOfDirection(i,j)] * d * (1 + 3*comp(i,j,v) + 4.5*sq(comp(i,j,v)) - 1.5*dot(v,v));
}

__kernel void collide_and_stream(__global       float* f_a,
                                 __global const float* f_b,
                                 __global       float* moments,
                                 __global const int* material)
{
    const unsigned int gid = indexOfCell(get_global_id(0), get_global_id(1));

    const uint2 cell = (uint2)(get_global_id(0), get_global_id(1));

    const int m = material[gid];

    if ( m == 0 ) {
        return;
    }

    const float  d = density(f_b, gid);
    const float2 v = velocity(f_b, d, gid);

    for ( int i = -1; i <= 1; ++i ) {
        for ( int j = 1; j >= -1; --j ) {
            *f_i(f_a, cell.x, cell.y, m*i, m*j) = *f_i(f_b, cell.x-i, cell.y-j, i, j)
                                                + $tau * (f_eq(d,v,i,j) - *f_i(f_b, cell.x-i, cell.y-j, i, j));
        }
    }

    moments[1*gid] = d;
    moments[2*gid] = v.x;
    moments[3*gid] = v.y;
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
                    self.np_material[self.idx(x,y)] = -1
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
            'tau': 0.56
        })).build()

    def evolve(self):
        if self.tick:
            self.tick = False
            self.program.collide_and_stream(self.queue, (self.nX,self.nY), (16,64), self.cl_pop_a, self.cl_pop_b, self.cl_moments, self.cl_material)
            self.queue.finish()
        else:
            self.tick = True
            self.program.collide_and_stream(self.queue, (self.nX,self.nY), (16,64), self.cl_pop_b, self.cl_pop_a, self.cl_moments, self.cl_material)
            self.queue.finish()

    def show(self, i):
        cl.enqueue_copy(LBM.queue, LBM.np_moments, LBM.cl_moments).wait();

        density = numpy.ndarray(shape=(self.nX-2, self.nY-2))
        for y in range(1,self.nY-1):
            for x in range(1,self.nX-1):
                density[x-1,y-1] = self.np_moments[0,self.idx(x,y)]

        plt.imshow(density, vmin=0.2, vmax=2.0, cmap=plt.get_cmap("seismic"))
        plt.savefig("result/density_" + str(i) + ".png")


def MLUPS(cells, steps, time):
    return ((cells*steps) / time) / 1000000

LBM = D2Q9_BGK_Lattice(1024, 1024)

nUpdates = 1000

start = timer()

for i in range(0,nUpdates):
    LBM.evolve()

end = timer()

runtime = end - start

print("Cells:   " + str(LBM.nCells))
print("Updates: " + str(nUpdates))
print("Time:    " + str(runtime))
print("MLUPS:   " + str(MLUPS(LBM.nCells, nUpdates, end - start)))

LBM.show(nUpdates)
