import pyopencl as cl
mf = cl.mem_flags

import numpy
import time

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')

from sympy import *
from sympy.codegen.ast import Assignment

from mako.template import Template

q = 9
d = 2

c = [ Matrix(x) for x in [(-1, 1), ( 0, 1), ( 1, 1), (-1, 0), ( 0, 0), ( 1, 0), (-1,-1), ( 0, -1), ( 1, -1)] ]
w = [ Rational(*x) for x in [(1,36), (1,9), (1,36), (1,9), (4,9), (1,9), (1,36), (1,9), (1,36)] ]

c_s = sqrt(Rational(1,3))

rho, tau = symbols('rho tau')

f_next = symarray('f_next', q)
f_curr = symarray('f_curr', q)

u = Matrix(symarray('u', d))

moments = [ Assignment(rho, sum(f_curr)) ]

for i, u_i in enumerate(u):
    moments.append(Assignment(u_i, sum([ (c_j*f_curr[j])[i] for j, c_j in enumerate(c) ]) / sum(f_curr)))

moments_opt = cse(moments, optimizations='basic', symbols=numbered_symbols(prefix='m'))

f_eq = []

for i, c_i in enumerate(c):
    f_eq_i = w[i] * rho * (  1
                           + c_i.dot(u)    /    c_s**2
                           + c_i.dot(u)**2 / (2*c_s**4)
                           - u.dot(u)      / (2*c_s**2) )
    f_eq.append(f_eq_i)

collide = [ Assignment(f_next[i], f_curr[i] + 1/tau * ( f_eq_i - f_curr[i] )) for i, f_eq_i in enumerate(f_eq) ]
collide_opt = cse(collide, optimizations='basic')

kernel = """
__constant float tau = ${tau};

unsigned int indexOfDirection(int i, int j) {
    return (i+1) + 3*(1-j);
}

unsigned int indexOfCell(int x, int y)
{
    return y * ${nX} + x;
}

unsigned int idx(int x, int y, int i, int j) {
    return indexOfDirection(i,j)*${nCells} + indexOfCell(x,y);
}

__global float f_i(__global __read_only float* f, int x, int y, int i, int j) {
    return f[idx(x,y,i,j)];
}

__kernel void collide_and_stream(__global __write_only float* f_a,
                                 __global __read_only  float* f_b,
                                 __global __read_only  int* material)
{
    const unsigned int gid = indexOfCell(get_global_id(0), get_global_id(1));

    const uint2 cell = (uint2)(get_global_id(0), get_global_id(1));

    const int m = material[gid];

    if ( m == 0 ) {
        return;
    }

% for i, c_i in enumerate(c):
    const float f_curr_${i} = f_i(f_b, cell.x-(${c_i[0]}), cell.y-(${c_i[1]}), ${c_i[0]}, ${c_i[1]});
% endfor

% for i, expr in enumerate(moments_helper):
    const float ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(moments_assignment):
    float ${ccode(expr)}
% endfor

    if ( m == 2 ) {
        u_0 = 0.0;
        u_1 = 0.0;
    }

% for i, expr in enumerate(collide_helper):
    const float ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(collide_assignment):
    const float ${ccode(expr)}
% endfor

% for i in range(0,len(c)):
    f_a[${i*nCells} + gid] = f_next_${i};
% endfor
}

__kernel void collect_moments(__global __read_only  float* f,
                              __global __write_only float* moments)
{
    const unsigned int gid = indexOfCell(get_global_id(0), get_global_id(1));

    const uint2 cell = (uint2)(get_global_id(0), get_global_id(1));

% for i in range(0,len(c)):
    const float f_curr_${i} = f[${i*nCells} + gid];
% endfor

% for i, expr in enumerate(moments_helper):
    const float ${expr[0]} = ${ccode(expr[1])};
% endfor

% for i, expr in enumerate(moments_assignment):
    moments[${i*nCells} + gid] = ${ccode(expr.rhs)};
% endfor
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

        self.np_stat_moments = []

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
        self.program = cl.Program(self.context, Template(kernel).render(
            nX     = self.nX,
            nY     = self.nY,
            nCells = self.nCells,
            tau    = '0.8f',
            moments_helper     = moments_opt[0],
            moments_assignment = moments_opt[1],
            collide_helper     = collide_opt[0],
            collide_assignment = collide_opt[1],
            c = c,
            ccode = ccode
        )).build()

    def collect_moments(self):
        if self.tick:
            self.program.collect_moments(self.queue, (self.nX,self.nY), (32,1), self.cl_pop_b, self.cl_moments)
        else:
            self.program.collect_moments(self.queue, (self.nX,self.nY), (32,1), self.cl_pop_a, self.cl_moments)

        cl.enqueue_copy(LBM.queue, self.np_moments, LBM.cl_moments).wait();
        self.np_stat_moments.append(self.np_moments.copy())

    def evolve(self):
        if self.tick:
            self.tick = False
            self.program.collide_and_stream(self.queue, (self.nX,self.nY), (32,1), self.cl_pop_a, self.cl_pop_b, self.cl_material)
        else:
            self.tick = True
            self.program.collide_and_stream(self.queue, (self.nX,self.nY), (32,1), self.cl_pop_b, self.cl_pop_a, self.cl_material)

    def sync(self):
        self.queue.finish()

    def generate_moment_plots(self):
        for i, np_moments in enumerate(self.np_stat_moments):
            print("Generating plot %d of %d." % (i+1, len(self.np_stat_moments)))

            density = numpy.ndarray(shape=(self.nX-2, self.nY-2))
            for y in range(1,self.nY-1):
                for x in range(1,self.nX-1):
                    density[y-1,x-1] = np_moments[0,self.idx(x,y)]

            plt.figure(figsize=(10, 10))
            plt.imshow(density, vmin=0.2, vmax=2.0, cmap=plt.get_cmap("seismic"))
            plt.savefig("result/density_" + str(i) + ".png", bbox_inches='tight', pad_inches=0)

        self.np_stat_moments = []


def MLUPS(cells, steps, time):
    return cells * steps / time * 1e-6

nUpdates = 1000
nStat = 100

print("Initializing simulation...\n")

LBM = D2Q9_BGK_Lattice(1024, 1024)

print("Starting simulation using %d cells...\n" % LBM.nCells)

lastStat = time.time()

for i in range(1,nUpdates+1):
    LBM.evolve()

    if i % nStat == 0:
        LBM.sync()
        print("i = %4d; %3.0f MLUPS" % (i, MLUPS(LBM.nCells, nStat, time.time() - lastStat)))
        LBM.collect_moments()
        lastStat = time.time()

print("\nConcluded simulation.\n")

LBM.generate_moment_plots()
