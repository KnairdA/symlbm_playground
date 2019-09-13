import numpy
from string import Template

from simulation         import Lattice, Geometry
from utility.particles  import Particles
from symbolic.generator import LBM

import symbolic.D3Q19 as D3Q19

from OpenGL.GL   import *
from OpenGL.GLUT import *

from OpenGL.GL import shaders

from pyrr import matrix44

lattice_x = 64
lattice_y = 32
lattice_z = 32

updates_per_frame = 20
particle_count = 50000

lid_speed = 0.001
relaxation_time = 0.515

def get_cavity_material_map(geometry):
    return [
        (lambda x, y, z: x > 0 and x < geometry.size_x-1 and
                         y > 0 and y < geometry.size_y-1 and
                         z > 0 and z < geometry.size_z-1,                                                1), # bulk fluid
        (lambda x, y, z: x == 1 or y == 1 or z == 1 or x == geometry.size_x-2 or y == geometry.size_y-2, 2), # walls
        (lambda x, y, z: z == geometry.size_z-2,                                                         3), # lid
        (lambda x, y, z: x == 0 or x == geometry.size_x-1 or
                         y == 0 or y == geometry.size_y-1 or
                         z == 0 or z == geometry.size_z-1,                                               0)  # ghost cells
    ]

boundary = """
    if ( m == 2 ) {
        u_0 = 0.0;
        u_1 = 0.0;
        u_2 = 0.0;
    }
    if ( m == 3 ) {
        u_0 = 0.1;
        u_1 = 0.0;
        u_2 = 0.0;
    }
"""

def get_projection(width, height):
    world_width = lattice_x
    world_height = world_width / width * height

    projection = matrix44.create_perspective_projection(45.0, width/height, 0.1, 1000.0)
    look = matrix44.create_look_at(
        eye    = [lattice_x/2, -2*lattice_y, lattice_z/2],
        target = [lattice_x/2, lattice_y/2, lattice_z/2],
        up     = [0, 0, 1])
    rotate = matrix44.create_from_axis_rotation(axis=[0,1,0], theta=0.2)

    point_size = 1

    return numpy.matmul(look, projection), point_size

def glut_window(fullscreen = False):
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    if fullscreen:
        window = glutEnterGameMode()
    else:
        glutInitWindowSize(800, 500)
        glutInitWindowPosition(0, 0)
        window = glutCreateWindow("LBM")

    return window

lbm = LBM(D3Q19)

window = glut_window(fullscreen = False)

vertex_shader = shaders.compileShader(Template("""
#version 430

layout (location=0) in vec4 particles;

out vec3 color;

uniform mat4 projection;

vec3 fire(float x) {
    return mix(
        vec3(1.0, 1.0, 0.0),
        vec3(1.0, 0.0, 0.0),
        x
    );
}

void main() {
    gl_Position = projection * vec4(
        particles[0],
        particles[1],
        particles[2],
        1.
    );

    color = fire(1.0-particles[3]);
}""").substitute({}), GL_VERTEX_SHADER)

fragment_shader = shaders.compileShader("""
#version 430

in vec3 color;

void main(){
    gl_FragColor = vec4(color.xyz, 0.0);
}""", GL_FRAGMENT_SHADER)

shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
projection_id = shaders.glGetUniformLocation(shader_program, 'projection')

lattice = Lattice(
    descriptor   = D3Q19,
    geometry     = Geometry(lattice_x, lattice_y, lattice_z),
    moments      = lbm.moments(optimize = True),
    collide      = lbm.bgk(f_eq = lbm.equilibrium(), tau = relaxation_time),
    boundary_src = boundary,
    opengl       = True
)

lattice.apply_material_map(
    get_cavity_material_map(lattice.geometry))
lattice.sync_material()

particles = Particles(
    lattice.context,
    lattice.queue,
    lattice.memory.float_type,
    numpy.mgrid[
        6*lattice.geometry.size_x//8:7*lattice.geometry.size_x//8:10j,
        lattice.geometry.size_y//8:7*lattice.geometry.size_y//8:particle_count/100j,
        6*lattice.geometry.size_z//8:7*lattice.geometry.size_z//8:10j,
    ].reshape(3,-1).T)

def on_display():
    for i in range(0,updates_per_frame):
        lattice.evolve()

    lattice.collect_gl_moments()

    for i in range(0,updates_per_frame):
        lattice.update_gl_particles(particles, aging = True)

    lattice.sync()

    glClear(GL_COLOR_BUFFER_BIT)
    glEnableClientState(GL_VERTEX_ARRAY)

    particles.gl_particles.bind()

    shaders.glUseProgram(shader_program)
    glUniformMatrix4fv(projection_id, 1, False, numpy.ascontiguousarray(projection))
    glVertexPointer(4, GL_FLOAT, 0, particles.gl_particles)
    glPointSize(point_size)
    glEnable(GL_POINT_SMOOTH)
    glDrawArrays(GL_POINTS, 0, particles.count)

    glutSwapBuffers()

def on_reshape(width, height):
    global projection, point_size
    glViewport(0,0,width,height)
    projection, point_size = get_projection(width, height)

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

glutDisplayFunc(on_display)
glutReshapeFunc(on_reshape)
glutTimerFunc(10, on_timer, 10)

glutMainLoop()
