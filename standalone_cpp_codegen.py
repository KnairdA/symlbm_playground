import sympy
from mako.template import Template
from pathlib import Path

from simulation         import Geometry
from symbolic.generator import LBM

import symbolic.D3Q19 as D3Q19

lbm = LBM(D3Q19)

moments = lbm.moments(optimize = True)
collide = lbm.bgk(f_eq = lbm.equilibrium(), tau = 0.6, optimize = True)

geometry = Geometry(64, 64, 64)

program_src = Template(filename = str(Path(__file__).parent/'template/standalone.mako')).render(
    descriptor = lbm.descriptor,
    geometry   = geometry,

    steps = 100,

    moments_subexpr    = moments[0],
    moments_assignment = moments[1],
    collide_subexpr    = collide[0],
    collide_assignment = collide[1],

    float_type = 'double',
    ccode = sympy.ccode
)

print(program_src)
