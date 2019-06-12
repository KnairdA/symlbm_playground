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

pop_eq = """
    if ( sqrt(pow(get_global_id(0) - ${nX//2}.f, 2.f) + pow(get_global_id(1) - ${nY//2}.f, 2.f)) < ${nX//10} ) {
% for i, w_i in enumerate(w):
        preshifted_f_a[${i*nCells}] = 1./24.f;
        preshifted_f_b[${i*nCells}] = 1./24.f;
% endfor
    } else {
% for i, w_i in enumerate(w):
        preshifted_f_a[${i*nCells}] = ${w_i}.f;
        preshifted_f_b[${i*nCells}] = ${w_i}.f;
% endfor
}"""

nUpdates = 2000
nStat    = 100

moments = []

print("Initializing simulation...\n")

lattice = Lattice(nX = 1024, nY = 1024, tau = 0.8, geometry = box, pop_eq_src = pop_eq)

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
