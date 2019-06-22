import numpy
import time

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from simulation         import Lattice, Geometry
from symbolic.generator import LBM

import symbolic.D3Q19 as D3Q19

from evtk.hl import imageToVTK

def MLUPS(cells, steps, time):
    return cells * steps / time * 1e-6

def export_vtk(lattice, moments):
    for i, m in enumerate(moments):
        print("Export VTK file %d of %d." % (i+1, len(moments)))

        imageToVTK("result/ldc_3d_%03d" % i, cellData = {
            'velocity_x': m[1,:].reshape(lattice.geometry.size(), order = 'F'),
            'velocity_y': m[2,:].reshape(lattice.geometry.size(), order = 'F'),
            'velocity_z': m[3,:].reshape(lattice.geometry.size(), order = 'F')
        })

def generate_moment_plots(lattice, moments):
    for i, m in enumerate(moments):
        print("Generating plot %d of %d." % (i+1, len(moments)))

        velocity = numpy.ndarray(shape=tuple(reversed(lattice.geometry.inner_size())))

        # extract x-z-plane
        y = lattice.geometry.size_y//2
        for z, x in numpy.ndindex(lattice.geometry.size_z-2, lattice.geometry.size_x-2):
            gid = lattice.gid(x+1,y,z+1)
            velocity[z,y,x] = numpy.sqrt(m[1,gid]**2 + m[2,gid]**2 + m[3,gid]**2)

        # extract y-z-plane
        x = lattice.geometry.size_x//2
        for z, y in numpy.ndindex(lattice.geometry.size_z-2, lattice.geometry.size_y-2):
            gid = lattice.gid(x,y+1,z+1)
            velocity[z,y,x] = numpy.sqrt(m[1,gid]**2 + m[2,gid]**2 + m[3,gid]**2)

        plt.figure(figsize=(20, 10))

        # plot x-z-plane
        plt.subplot(1, 2, 1)
        plt.imshow(velocity[:,y,:], origin='lower', vmin=0.0, vmax=0.15, cmap=plt.get_cmap('seismic'))

        # plot y-z-plane
        plt.subplot(1, 2, 2)
        plt.imshow(velocity[:,:,x], origin='lower', vmin=0.0, vmax=0.15, cmap=plt.get_cmap('seismic'))

        plt.savefig("result/ldc_3d_%02d.png" % i, bbox_inches='tight', pad_inches=0)

def cavity(geometry, x, y, z):
    if x == 1 or y == 1 or z == 1 or x == geometry.size_x-2 or y == geometry.size_y-2:
        return 2
    elif z == geometry.size_z-2:
        return 3
    else:
        return 1

boundary = """
    if ( m == 2 ) {
        u_0 = 0.0;
        u_1 = 0.0;
        u_2 = 0.0;
    }
    if ( m == 3 ) {
        u_0 = 0.1;
        u_1 = 0.0;
        u_2 = 0.0;
    }
"""

nUpdates = 10000
nStat    = 1000

moments = []

print("Initializing simulation...\n")

lbm = LBM(D3Q19)

lattice = Lattice(
    descriptor = D3Q19,
    geometry   = Geometry(128, 128, 128),

    moments = lbm.moments(optimize = False),
    collide = lbm.bgk(f_eq = lbm.equilibrium(), tau = 0.56),

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

#export_vtk(lattice, moments)
generate_moment_plots(lattice, moments)
