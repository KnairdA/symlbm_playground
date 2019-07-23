import sympy
from mako.template import Template
from pathlib import Path

from simulation         import Geometry
from symbolic.generator import LBM

import symbolic.D2Q9 as D2Q9

lbm = LBM(D2Q9)

moments = lbm.moments(optimize = False)
collide = lbm.bgk(f_eq = lbm.equilibrium(), tau = 0.6)

geometry = Geometry(512, 512)

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
