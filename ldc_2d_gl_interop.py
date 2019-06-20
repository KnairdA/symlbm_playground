import numpy
import time

from simulation         import Lattice, Geometry
from symbolic.generator import LBM

import symbolic.D2Q9 as D2Q9

from OpenGL.GL   import *
from OpenGL.GLUT import *

screen_x = 1920
screen_y = 1200

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

def glut_window(fullscreen = False):
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    if fullscreen:
        window = glutEnterGameMode()
    else:
        glutInitWindowSize(screen_x, screen_y)
        glutInitWindowPosition(0, 0)
        window = glutCreateWindow("LBM")

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    glOrtho(
        0, screen_x,
        0, screen_y,
        0.1, 100.0
    )

    return window

lbm = LBM(D2Q9)

window = glut_window(fullscreen = False)

lattice = Lattice(
    descriptor   = D2Q9,
    geometry     = Geometry(screen_x//4, screen_y//4),
    moments      = lbm.moments(optimize = False),
    collide      = lbm.bgk(f_eq = lbm.equilibrium(), tau = 0.52),
    boundary_src = boundary,
    opengl       = True
)

lattice.setup_geometry(cavity)

def on_display():
    for i in range(0,100):
        lattice.evolve()

    lattice.sync_gl_moments()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glTranslatef(0., 0., -1.)

    lattice.gl_moments.bind()
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(4, GL_FLOAT, 0, lattice.gl_moments)
    glDrawArrays(GL_POINTS, 0, lattice.geometry.volume)
    glDisableClientState(GL_VERTEX_ARRAY)

    glutSwapBuffers()

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

glutDisplayFunc(on_display)
glutTimerFunc(10, on_timer, 10)

glutMainLoop()
