import numpy
from string import Template

from simulation         import Lattice, Geometry
from symbolic.generator import LBM

import symbolic.D2Q9 as D2Q9

from OpenGL.GL   import *
from OpenGL.GLUT import *

from OpenGL.GL import shaders

screen_x = 1920
screen_y = 1200
pixels_per_cell   = 2
updates_per_frame = 80

inflow = 0.02
relaxation_time = 0.51

def get_obstacles(geometry):
    ys = numpy.linspace(geometry.size_y//50, geometry.size_y-geometry.size_y//50, num = 20)
    xs = [ 50 for i, y in enumerate(ys) ]
    return list(zip(xs, ys))

def is_obstacle(geometry, x, y):
    for (ox,oy) in obstacles:
        if numpy.sqrt((x - ox)**2 + (y - oy)**2) < geometry.size_x//100:
            return True

    else:
        return False

def channel(geometry, x, y):
    if x == 1:
        return 3
    elif x == geometry.size_x-2:
        return 4
    elif y == 1 or y == geometry.size_y-2:
        return 2
    elif is_obstacle(geometry, x, y):
        return 2
    else:
        return 1

boundary = Template("""
    if ( m == 2 ) {
        u_0 = 0.0;
        u_1 = 0.0;
    }
    if ( m == 3 ) {
        u_0 = min(time/10000.0 * $inflow, $inflow);
        u_1 = 0.0;
    }
    if ( m == 4 ) {
        rho = 1.0;
    }
""").substitute({
    'inflow': inflow
})

def get_projection():
    scale = numpy.diag([(2.0*pixels_per_cell)/screen_x, (2.0*pixels_per_cell)/screen_y, 1.0, 1.0])
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

out vec3 color;

uniform mat4 projection;

vec3 blueRedPalette(float x) {
    return mix(
        vec3(0.0, 0.0, 1.0),
        vec3(1.0, 0.0, 0.0),
        x
    );
}

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
        idx.x,
        idx.y,
        0.,
        1.
    );

    color = blueRedPalette(CellMoments[3] / 0.08);
}""").substitute({
    'size_x': screen_x//pixels_per_cell,
    'inflow': inflow
}), GL_VERTEX_SHADER)

fragment_shader = shaders.compileShader("""
#version 430

in vec3 color;

void main(){
    gl_FragColor = vec4(color.xyz, 0.0);
}""", GL_FRAGMENT_SHADER)


shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
projection_id = shaders.glGetUniformLocation(shader_program, 'projection')

lattice = Lattice(
    descriptor   = D2Q9,
    geometry     = Geometry(screen_x//pixels_per_cell, screen_y//pixels_per_cell),
    moments      = lbm.moments(optimize = False),
    collide      = lbm.bgk(f_eq = lbm.equilibrium(), tau = relaxation_time),
    boundary_src = boundary,
    opengl       = True
)

obstacles = get_obstacles(lattice.geometry)
lattice.setup_geometry(channel)

projection = get_projection()

def on_display():
    for i in range(0,updates_per_frame):
        lattice.evolve()

    lattice.sync_gl_moments()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    lattice.memory.gl_moments.bind()
    glEnableClientState(GL_VERTEX_ARRAY)

    shaders.glUseProgram(shader_program)
    glUniformMatrix4fv(projection_id, 1, False, numpy.asfortranarray(projection))

    glVertexPointer(4, GL_FLOAT, 0, lattice.memory.gl_moments)

    glPointSize(pixels_per_cell)
    glDrawArrays(GL_POINTS, 0, lattice.geometry.volume)

    glDisableClientState(GL_VERTEX_ARRAY)

    glutSwapBuffers()

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

glutDisplayFunc(on_display)
glutTimerFunc(10, on_timer, 10)

glutMainLoop()
