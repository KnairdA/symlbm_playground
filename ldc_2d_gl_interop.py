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
cells_per_pixel   = 4
updates_per_frame = 200

lid_speed = 0.1
relaxation_time = 0.515

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

def get_projection():
    scale = numpy.diag([(2.0*cells_per_pixel)/screen_x, (2.0*cells_per_pixel)/screen_y, 1.0, 1.0])
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

    color = blueRedPalette(CellMoments[3] / $lid_speed);
}""").substitute({
    'size_x'   : screen_x//cells_per_pixel,
    'lid_speed': lid_speed
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
    geometry     = Geometry(screen_x//cells_per_pixel, screen_y//cells_per_pixel),
    moments      = lbm.moments(optimize = True),
    collide      = lbm.bgk(f_eq = lbm.equilibrium(), tau = relaxation_time),
    boundary_src = boundary,
    opengl       = True
)

lattice.setup_geometry(cavity)

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

    glPointSize(cells_per_pixel)
    glDrawArrays(GL_POINTS, 0, lattice.geometry.volume)

    glDisableClientState(GL_VERTEX_ARRAY)

    glutSwapBuffers()

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

glutDisplayFunc(on_display)
glutTimerFunc(10, on_timer, 10)

glutMainLoop()
