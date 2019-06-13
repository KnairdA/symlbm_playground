import numpy
import time

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')

from D2Q9 import Lattice

def MLUPS(cells, steps, time):
    return cells * steps / time * 1e-6

def generate_moment_plots(lattice, moments):
    for i, m in enumerate(moments):
        print("Generating plot %d of %d." % (i+1, len(moments)))

        velocity = numpy.ndarray(shape=(lattice.nY-2, lattice.nX-2))
        for y in range(1,lattice.nY-1):
            for x in range(1,lattice.nX-1):
                velocity[y-1,x-1] = numpy.sqrt(m[1,lattice.idx(x,y)]**2 + m[2,lattice.idx(x,y)]**2)

        plt.figure(figsize=(10, 10))
        plt.imshow(velocity, origin='lower', cmap=plt.get_cmap('seismic'))
        plt.savefig("result/velocity_" + str(i) + ".png", bbox_inches='tight', pad_inches=0)

def box(nX, nY, x, y):
    if x == 1 or y == 1 or x == nX-2:
        return 2
    elif y == nY-2:
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

lattice = Lattice(nX = 256, nY = 256, tau = 0.56, geometry = box, boundary_src = boundary)

print("Starting simulation using %d cells...\n" % lattice.nCells)

lastStat = time.time()

for i in range(1,nUpdates+1):
    lattice.evolve()

    if i % nStat == 0:
        lattice.sync()
        print("i = %4d; %3.0f MLUPS" % (i, MLUPS(lattice.nCells, nStat, time.time() - lastStat)))
        moments.append(lattice.get_moments())
        lastStat = time.time()

print("\nConcluded simulation.\n")

generate_moment_plots(lattice, moments)
