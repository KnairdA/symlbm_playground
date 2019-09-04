import numpy
from string import Template

from simulation         import Lattice, Geometry
from utility.particles  import Particles
from symbolic.generator import LBM

import symbolic.D2Q9 as D2Q9

from OpenGL.GL   import *
from OpenGL.GLUT import *

from OpenGL.GL import shaders

screen_x = 1920
screen_y = 1200
pixels_per_cell   = 4
updates_per_frame = 100

inflow = 0.01
relaxation_time = 0.52

def circle(cx, cy, r):
    return lambda x, y: (x - cx)**2 + (y - cy)**2 < r*r

def get_channel_material_map(geometry):
    return [
        (lambda x, y: x > 0 and x < geometry.size_x-1 and y > 0 and y < geometry.size_y-1, 1), # bulk fluid
        (lambda x, y: x == 1,                 3), # inflow
        (lambda x, y: x == geometry.size_x-2, 4), # outflow
        (lambda x, y: y == 1,                 2), # bottom
        (lambda x, y: y == geometry.size_y-2, 2), # top
        (lambda x, y: x > geometry.size_x//20 and x < 2*geometry.size_x//20 and y < 4*geometry.size_y//9, 2),
        (lambda x, y: x > geometry.size_x//20 and x < 2*geometry.size_x//20 and y > 5*geometry.size_y//9, 2),
        (circle(geometry.size_x//4,    geometry.size_y//2, 50), 2),
        (circle(geometry.size_x//4-25, geometry.size_y//2, 50), 1),
        (lambda x, y: x == 0 or x == geometry.size_x-1 or y == 0 or y == geometry.size_y-1, 0) # ghost cells
    ]

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

    if (CellMoments[3] > 0.0) {
        color = blueRedPalette(CellMoments[3] / 0.2);
    } else {
        color = vec3(0.0,0.0,0.0);
    }
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

particle_shader = shaders.compileShader(Template("""
#version 430

layout (location=0) in vec4 Particles;

out vec3 color;

uniform mat4 projection;

void main() {
    gl_Position = projection * vec4(
        Particles[0],
        Particles[1],
        0.,
        1.
    );

    color = vec3(0.0,0.8,0.0);
}""").substitute({}), GL_VERTEX_SHADER)

shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
particle_program = shaders.compileProgram(particle_shader, fragment_shader)
projection_id = shaders.glGetUniformLocation(shader_program, 'projection')

lattice = Lattice(
    descriptor   = D2Q9,
    geometry     = Geometry(screen_x//pixels_per_cell, screen_y//pixels_per_cell),
    moments      = lbm.moments(optimize = False),
    collide      = lbm.bgk(f_eq = lbm.equilibrium(), tau = relaxation_time),
    boundary_src = boundary,
    opengl       = True
)

lattice.apply_material_map(
    get_channel_material_map(lattice.geometry))
lattice.sync_material()

projection = get_projection()

particles = Particles(
    lattice.context,
    lattice.memory.float_type,
    numpy.mgrid[
        lattice.geometry.size_x//20:2*lattice.geometry.size_x//20:100j,
        4*lattice.geometry.size_y//9:5*lattice.geometry.size_y//9:100000/100j
    ].reshape(2,-1).T)

def on_display():
    for i in range(0,updates_per_frame):
        lattice.evolve()
        lattice.update_gl_particles(particles)

    lattice.sync()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    lattice.memory.gl_moments.bind()
    glEnableClientState(GL_VERTEX_ARRAY)

    shaders.glUseProgram(shader_program)
    glUniformMatrix4fv(projection_id, 1, False, numpy.asfortranarray(projection))

    glVertexPointer(4, GL_FLOAT, 0, lattice.memory.gl_moments)

    glDisable(GL_POINT_SMOOTH)
    glPointSize(pixels_per_cell)
    glDrawArrays(GL_POINTS, 0, lattice.geometry.volume)

    particles.gl_particles.bind()
    glEnableClientState(GL_VERTEX_ARRAY)

    shaders.glUseProgram(particle_program)
    glUniformMatrix4fv(projection_id, 1, False, numpy.asfortranarray(projection))

    glVertexPointer(4, GL_FLOAT, 0, particles.gl_particles)

    glEnable(GL_POINT_SMOOTH)
    glPointSize(1)
    glDrawArrays(GL_POINTS, 0, particles.count)

    glDisableClientState(GL_VERTEX_ARRAY)

    glutSwapBuffers()

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

glutDisplayFunc(on_display)
glutTimerFunc(10, on_timer, 10)

glutMainLoop()
