import numpy
import time
from string import Template

from simulation         import Lattice, Geometry
from symbolic.generator import LBM

import symbolic.D2Q9 as D2Q9

from OpenGL.GL   import *
from OpenGL.GLUT import *

from OpenGL.GL import shaders

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

def get_projection():
    scale = numpy.diag([8.0/screen_x, 8.0/screen_y, 1.0, 1.0])
    translation        = numpy.matrix(numpy.identity(4))
    translation[3,0:3] = [-1.0, -1.0, 0.0]
    return scale * translation;

def glut_window(fullscreen = False):
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    if fullscreen:
        window = glutEnterGameMode()
    else:
        glutInitWindowSize(screen_x, screen_y)
        glutInitWindowPosition(0, 0)
        window = glutCreateWindow("LBM")

    return window

lbm = LBM(D2Q9)

window = glut_window(fullscreen = False)

vertex_shader = shaders.compileShader(Template("""
#version 430

layout (location=0) in vec4 CellMoments;

uniform mat4 projection;

vec2 fluidVertexAtIndex(uint i) {
    const float y = floor(float(i) / $size_x);
	return vec2(
		i - $size_x*y,
		y
	);
}

void main() {
    const vec2 idx = fluidVertexAtIndex(gl_VertexID);

    gl_Position = projection * vec4(
        idx.x + 500.*CellMoments[1],
        idx.y + 500.*CellMoments[2],
        0.,
        1.
    );
}""").substitute({'size_x': screen_x//4}), GL_VERTEX_SHADER)

fragment_shader = shaders.compileShader("""
#version 120
void main(){
    gl_FragColor = vec4(1,1,1,1);
}""", GL_FRAGMENT_SHADER)



shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
projection_id = shaders.glGetUniformLocation(shader_program, 'projection')

lattice = Lattice(
    descriptor   = D2Q9,
    geometry     = Geometry(screen_x//4, screen_y//4),
    moments      = lbm.moments(optimize = True),
    collide      = lbm.bgk(f_eq = lbm.equilibrium(), tau = 0.515),
    boundary_src = boundary,
    opengl       = True
)

lattice.setup_geometry(cavity)

projection = get_projection()

def on_display():
    for i in range(0,60):
        lattice.evolve()

    lattice.sync_gl_moments()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    lattice.gl_moments.bind()
    glEnableClientState(GL_VERTEX_ARRAY)

    shaders.glUseProgram(shader_program)
    glUniformMatrix4fv(projection_id, 1, False, numpy.asfortranarray(projection))

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
