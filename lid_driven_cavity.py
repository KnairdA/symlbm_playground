import numpy
import time

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')

from lbm import Lattice, Geometry

import symbolic.D2Q9 as D2Q9

def MLUPS(cells, steps, time):
    return cells * steps / time * 1e-6

def generate_moment_plots(lattice, moments):
    for i, m in enumerate(moments):
        print("Generating plot %d of %d." % (i+1, len(moments)))

        velocity = numpy.ndarray(shape=tuple(reversed(lattice.geometry.inner_span())))
        for x, y in lattice.geometry.inner_cells():
            velocity[y-1,x-1] = numpy.sqrt(m[1,lattice.idx(x,y)]**2 + m[2,lattice.idx(x,y)]**2)

        plt.figure(figsize=(10, 10))
        plt.imshow(velocity, origin='lower', cmap=plt.get_cmap('seismic'))
        plt.savefig("result/ldc_" + str(i) + ".png", bbox_inches='tight', pad_inches=0)

def cavity(geometry, x, y):
    if x == 1 or y == 1 or x == geometry.size_x-2:
        return 2
    elif y == geometry.size_y-2:
        return 3
    else:
        return 1

boundary = """
    if ( m == 2 ) {
        u_0 = 0.0;
        u_1 = 0.0;
    }
    if ( m == 3 ) {
        u_0 = 0.1;
        u_1 = 0.0;
    }
"""

nUpdates = 100000
nStat    = 5000

moments = []

print("Initializing simulation...\n")

lattice = Lattice(
    descriptor = D2Q9,
    geometry   = Geometry(256, 256),

    moments = D2Q9.moments(optimize = False),
    collide = D2Q9.bgk(tau = 0.56),

    boundary_src = boundary)

lattice.setup_geometry(cavity)

print("Starting simulation using %d cells...\n" % lattice.geometry.volume)

lastStat = time.time()

for i in range(1,nUpdates+1):
    lattice.evolve()

    if i % nStat == 0:
        lattice.sync()
        print("i = %4d; %3.0f MLUPS" % (i, MLUPS(lattice.geometry.volume, nStat, time.time() - lastStat)))
        moments.append(lattice.get_moments())
        lastStat = time.time()

print("\nConcluded simulation.\n")

generate_moment_plots(lattice, moments)
