import numpy
import time
from string import Template

from simulation         import Lattice, Geometry
from symbolic.generator import LBM

import symbolic.D3Q19 as D3Q19
import symbolic.D3Q27 as D3Q27

import itertools

lid_speed = 0.1
relaxation_time = 0.52

def MLUPS(cells, steps, time):
    return cells * steps / time * 1e-6

def cavity(geometry, x, y, z):
    if x == 1 or y == 1 or z == 1 or x == geometry.size_x-2 or y == geometry.size_y-2:
        return 2
    elif z == geometry.size_z-2:
        return 3
    else:
        return 1

boundary = Template("""
    if ( m == 2 ) {
        u_0 = 0.0;
        u_1 = 0.0;
        u_2 = 0.0;
    }
    if ( m == 3 ) {
        u_0 = $lid_speed;
        u_1 = 0.0;
        u_2 = 0.0;
    }
""").substitute({
    'lid_speed': lid_speed
})

base_2_sizes  = {16, 32, 48, 64, 96, 128}
base_10_sizes = {10, 20, 40, 60, 80, 100}

base_2_layouts = {
    (  16, 1, 1),
    (  32, 1, 1),
    (  64, 1, 1),
    (  96, 1, 1),
    ( 128, 1, 1),
}

base_10_layouts = {
    (  10, 1, 1),
    (  20, 1, 1),
    (  30, 1, 1),
    (  50, 1, 1),
    ( 100, 1, 1)
}

descriptors = { D3Q19, D3Q27 }

precisions = { 'single', 'double' }

base_2_configs = list(filter(
    lambda config: config[0] % config[1][0] == 0,
    itertools.product(*[base_2_sizes, base_2_layouts, descriptors, precisions, {True, False}, {True}])
))

align_configs = list(filter(
    lambda config: config[0] % config[1][0] == 0,
    itertools.product(*[base_10_sizes, base_10_layouts, descriptors, precisions, {True, False}, {True, False}])
))

pad_configs = list(filter(
    lambda config: config[0] - config[1][0] >= -28,
    itertools.product(*[base_10_sizes, base_2_layouts, descriptors, precisions, {True, False}, {True}])
))

measurements = []

for size, layout, descriptor, precision, opti, align in base_2_configs + align_configs + pad_configs:
    lbm = LBM(descriptor)
    lattice = Lattice(
        descriptor = descriptor,
        geometry   = Geometry(size, size, size),
        precision = precision,
        layout  = layout,
        padding = layout,
        align   = align,
        moments = lbm.moments(optimize = opti),
        collide = lbm.bgk(f_eq = lbm.equilibrium(), tau = relaxation_time, optimize = opti),
        boundary_src = boundary)
    lattice.setup_geometry(cavity)

    nUpdates = 500
    nStat = 100

    stats = []

    lastStat = time.time()

    for i in range(1,nUpdates+1):
        lattice.evolve()

        if i % nStat == 0:
            lattice.sync()
            mlups = round(MLUPS(lattice.geometry.volume, nStat, time.time() - lastStat))
            stats.append(mlups)
            lastStat = time.time()

    print('%s: ~%d MLUPS' % ((size, layout, descriptor.__name__, precision, opti, align), numpy.average(stats)))
    measurements.append(((size, layout, descriptor.__name__, precision, opti, align), stats))
    del lattice, lbm

with open('result/ldc_3d_benchmark.data', 'w') as f:
    f.write(str(measurements))
