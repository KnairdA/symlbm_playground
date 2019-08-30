import numpy
import time
from string import Template

from simulation         import Lattice, Geometry
from symbolic.generator import LBM

import symbolic.D2Q9 as D2Q9

import itertools

lid_speed = 0.1
relaxation_time = 0.52

def MLUPS(cells, steps, time):
    return cells * steps / time * 1e-6

def get_cavity_material_map(geometry):
    return [
        (lambda x, y: x > 0 and x < geometry.size_x-1 and y > 0 and y < geometry.size_y-1,  1), # bulk fluid
        (lambda x, y: x == 1 or y == 1 or x == geometry.size_x-2,                           2), # left, right, bottom walls
        (lambda x, y: y == geometry.size_y-2,                                               3), # lid
        (lambda x, y: x == 0 or x == geometry.size_x-1 or y == 0 or y == geometry.size_y-1, 0)  # ghost cells
    ]

boundary = Template("""
    if ( m == 2 ) {
        u_0 = 0.0;
        u_1 = 0.0;
    }
    if ( m == 3 ) {
        u_0 = $lid_speed;
        u_1 = 0.0;
    }
""").substitute({
    'lid_speed': lid_speed
})

base_2_sizes  = {32, 64, 128, 256, 512, 1024, 2048}
base_10_sizes = {50, 100, 200, 400, 600, 800, 1000}

precisions = {'single', 'double'}

base_2_layouts = {
    (  16, 1),
    (  32, 1),
    (  64, 1),
    ( 128, 1),
    ( 256, 1),
    ( 512, 1),
    (1024, 1),
}

base_10_layouts = {
    (  10, 1),
    (  30, 1),
    (  50, 1),
    ( 100, 1),
    ( 200, 1)
}

base_2_configs = list(filter(
    lambda config: config[0] % config[1][0] == 0,
    itertools.product(*[base_2_sizes, base_2_layouts, precisions, {True, False}, {True}])
))

align_configs = list(filter(
    lambda config: config[0] % config[1][0] == 0,
    itertools.product(*[base_10_sizes, base_10_layouts, precisions, {True, False}, {True, False}])
))

pad_configs = list(filter(
    lambda config: config[0] - config[1][0] >= -100,
    itertools.product(*[base_10_sizes, base_2_layouts, precisions, {True, False}, {True}])
))

lbm = LBM(D2Q9)

measurements = []

for size, layout, precision, opti, align in base_2_configs + align_configs + pad_configs:
    lattice = Lattice(
        descriptor = D2Q9,
        geometry   = Geometry(size, size),
        precision = precision,
        layout  = layout,
        padding = layout,
        align   = align,
        moments = lbm.moments(optimize = opti),
        collide = lbm.bgk(f_eq = lbm.equilibrium(), tau = relaxation_time, optimize = opti),
        boundary_src = boundary)
    lattice.apply_material_map(
        get_cavity_material_map(lattice.geometry))
    lattice.sync_material()

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

    print('%s: ~%d MLUPS' % ((size, layout, precision, opti, align), numpy.average(stats)))
    measurements.append(((size, layout, precision, opti, align), stats))
    del lattice

with open('result/ldc_2d_benchmark.data', 'w') as f:
    f.write(str(measurements))
