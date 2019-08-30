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
        y_slice = lattice.geometry.size_y//2
        for z, x in numpy.ndindex(lattice.geometry.size_z-2, lattice.geometry.size_x-2):
            gid = lattice.memory.gid(x+1,y_slice,z+1)
            velocity[z,y_slice,x] = numpy.sqrt(m[1,gid]**2 + m[2,gid]**2 + m[3,gid]**2)

        # extract y-z-plane
        x_slice = lattice.geometry.size_x//2
        for z, y in numpy.ndindex(lattice.geometry.size_z-2, lattice.geometry.size_y-2):
            gid = lattice.memory.gid(x_slice,y+1,z+1)
            velocity[z,y,x_slice] = numpy.sqrt(m[1,gid]**2 + m[2,gid]**2 + m[3,gid]**2)

        plt.figure(figsize=(20, 10))

        # plot x-z-plane
        plt.subplot(1, 2, 1)
        plt.imshow(velocity[:,y_slice,:], origin='lower', vmin=0.0, vmax=0.15, cmap=plt.get_cmap('seismic'))

        # plot y-z-plane
        plt.subplot(1, 2, 2)
        plt.imshow(velocity[:,:,x_slice], origin='lower', vmin=0.0, vmax=0.15, cmap=plt.get_cmap('seismic'))

        plt.savefig("result/ldc_3d_%02d.png" % i, bbox_inches='tight', pad_inches=0)

def get_cavity_material_map(geometry):
    return [
        (lambda x, y, z: x > 0 and x < geometry.size_x-1 and
                         y > 0 and y < geometry.size_y-1 and
                         z > 0 and z < geometry.size_z-1,                                                1), # bulk fluid
        (lambda x, y, z: x == 1 or y == 1 or z == 1 or x == geometry.size_x-2 or y == geometry.size_y-2, 2), # walls
        (lambda x, y, z: z == geometry.size_z-2,                                                         3), # lid
        (lambda x, y, z: x == 0 or x == geometry.size_x-1 or
                         y == 0 or y == geometry.size_y-1 or
                         z == 0 or z == geometry.size_z-1,                                               0)  # ghost cells
    ]

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
    geometry   = Geometry(64, 64, 64),

    moments = lbm.moments(optimize = False),
    collide = lbm.bgk(f_eq = lbm.equilibrium(), tau = 0.56),

    boundary_src = boundary)

lattice.apply_material_map(
    get_cavity_material_map(lattice.geometry))
lattice.sync_material()

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
