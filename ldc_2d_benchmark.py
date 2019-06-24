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

def cavity(geometry, x, y):
    if x == 1 or y == 1 or x == geometry.size_x-2:
        return 2
    elif y == geometry.size_y-2:
        return 3
    else:
        return 1

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

sizes = [32, 64, 96, 128, 256, 512, 1024]

precisions = [ 'single', 'double' ]

layouts = [
    (  16, 1),
    (  24, 1),
    (  32, 1),
    (  48, 1),
    (  64, 1),
    (  96, 1),
    ( 128, 1),
    ( 256, 1),
    ( 512, 1),
    (1024, 1),
]

configs = list(filter(
    lambda config: config[0] % config[1][0] == 0 and config[0] % config[1][1] == 0,
    itertools.product(*[sizes, layouts, precisions, [True, False]])
))

lbm = LBM(D2Q9)

measurements = []

for size, layout, precision, opti in configs:
    lattice = Lattice(
        descriptor = D2Q9,
        geometry   = Geometry(size, size),
        precision = precision,
        layout  = layout,
        moments = lbm.moments(optimize = opti),
        collide = lbm.bgk(f_eq = lbm.equilibrium(), tau = relaxation_time, optimize = opti),
        boundary_src = boundary)
    lattice.setup_geometry(cavity)

    nUpdates = 1000
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

    print('%s: ~%d MLUPS' % ((size, layout, precision, opti), numpy.average(stats)))
    measurements.append(((size, layout, precision, opti), stats))
    del lattice

with open('result/ldc_2d_benchmark.data', 'w') as f:
    f.write(str(measurements))
