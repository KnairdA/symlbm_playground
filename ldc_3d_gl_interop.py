import numpy
from string import Template

from simulation         import Lattice, Geometry
from utility.particles  import Particles
from symbolic.generator import LBM

import symbolic.D3Q19 as D3Q19

from OpenGL.GL   import *
from OpenGL.GLUT import *

from OpenGL.GL import shaders

from pyrr import matrix44, quaternion

lattice_x = 64
lattice_y = 64
lattice_z = 64

updates_per_frame = 20
particle_count = 50000

lid_speed = 0.1
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

boundary = Template("""
    if ( m == 2 ) {
        u_0 = 0.0;
        u_1 = 0.0;
        u_2 = 0.0;
    }
    if ( m == 3 ) {
        u_0 = $lid_speed;
        u_1 = 0.0;
        u_2 = 0.0;
    }
""").substitute({
    "lid_speed": lid_speed
})

def get_projection(width, height):
    world_width = lattice_x
    world_height = world_width / width * height

    projection = matrix44.create_perspective_projection(45.0, width/height, 0.1, 1000.0)
    look = matrix44.create_look_at(
        eye    = [0, -2*lattice_y, 0],
        target = [0, 0, 0],
        up     = [0, 0, -1])

    point_size = 1

    return numpy.matmul(look, projection), point_size

class Rotation:
    def __init__(self, shift, x = 0.0, z = 0.0):
        self.shift = shift
        self.rotation_x = x
        self.rotation_z = z

    def update(self, x, z):
        self.rotation_x += x
        self.rotation_z += z

    def get(self):
        qx = quaternion.Quaternion(quaternion.create(1,0,0,self.rotation_x))
        qz = quaternion.Quaternion(quaternion.create(0,0,1,self.rotation_z))
        rotation = qz.cross(qx)
        return numpy.matmul(
            matrix44.create_from_translation(self.shift),
            matrix44.create_from_quaternion(rotation)
        )

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

particle_shader = shaders.compileShader(Template("""
#version 430

layout (location=0) in vec4 particles;
                   out vec3 color;

uniform mat4 projection;
uniform mat4 rotation;

vec3 fire(float x) {
    return mix(
        vec3(1.0, 1.0, 0.0),
        vec3(1.0, 0.0, 0.0),
        x
    );
}

void main() {
    gl_Position = projection * rotation * vec4(
        particles[0],
        particles[1],
        particles[2],
        1.
    );

    color = fire(1.0-particles[3]);
}""").substitute({}), GL_VERTEX_SHADER)

vertex_shader = shaders.compileShader(Template("""
#version 430

layout (location=0) in vec4 vertex;
                   out vec3 color;

uniform mat4 projection;
uniform mat4 rotation;

void main() {
    gl_Position = projection * rotation * vertex;
    color = vec3(1.0,1.0,1.0);
}""").substitute({}), GL_VERTEX_SHADER)

fragment_shader = shaders.compileShader("""
#version 430

in vec3 color;

void main(){
    gl_FragColor = vec4(color.xyz, 0.0);
}""", GL_FRAGMENT_SHADER)

particle_program = shaders.compileProgram(particle_shader, fragment_shader)
projection_id = shaders.glGetUniformLocation(particle_program, 'projection')
rotation_id   = shaders.glGetUniformLocation(particle_program, 'rotation')

geometry_program = shaders.compileProgram(vertex_shader, fragment_shader)

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
        8*lattice.geometry.size_x//10:9*lattice.geometry.size_x//10:10j,
        lattice.geometry.size_y//10:9*lattice.geometry.size_y//10:particle_count/100j,
        8*lattice.geometry.size_z//10:9*lattice.geometry.size_z//10:10j,
    ].reshape(3,-1).T)

rotation = Rotation([-lattice_x/2, -lattice_y/2, -lattice_z/2])

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

    shaders.glUseProgram(particle_program)
    glUniformMatrix4fv(projection_id, 1, False, numpy.ascontiguousarray(projection))
    glUniformMatrix4fv(rotation_id, 1, False, numpy.ascontiguousarray(rotation.get()))
    glVertexPointer(4, GL_FLOAT, 0, particles.gl_particles)
    glPointSize(point_size)
    glDrawArrays(GL_POINTS, 0, particles.count)

    shaders.glUseProgram(geometry_program)
    glUniformMatrix4fv(projection_id, 1, False, numpy.ascontiguousarray(projection))
    glUniformMatrix4fv(rotation_id, 1, False, numpy.ascontiguousarray(rotation.get()))
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glBegin(GL_POLYGON)
    glVertex3f(0,0,0)
    glVertex3f(lattice_x,0,0)
    glVertex3f(lattice_x,lattice_y,0)
    glVertex3f(0,lattice_y,0)
    glEnd()
    glBegin(GL_POLYGON)
    glVertex3f(0,0,lattice_z)
    glVertex3f(lattice_x,0,lattice_z)
    glVertex3f(lattice_x,lattice_y,lattice_z)
    glVertex3f(0,lattice_y,lattice_z)
    glEnd()
    glBegin(GL_LINES)
    glVertex3f(0,0,0)
    glVertex3f(0,0,lattice_z)
    glVertex3f(lattice_x,0,0)
    glVertex3f(lattice_x,0,lattice_z)
    glVertex3f(lattice_x,lattice_y,0)
    glVertex3f(lattice_x,lattice_y,lattice_z)
    glVertex3f(0,lattice_y,0)
    glVertex3f(0,lattice_y,lattice_z)
    glEnd()

    glutSwapBuffers()

def on_reshape(width, height):
    global projection, point_size
    glViewport(0,0,width,height)
    projection, point_size = get_projection(width, height)

def on_keyboard(key, x, y):
    global rotation

    x = {
        b'w': -numpy.pi/100,
        b's':  numpy.pi/100
    }.get(key, 0.0)
    z = {
        b'a': -numpy.pi/100,
        b'd':  numpy.pi/100
    }.get(key, 0.0)

    rotation.update(x,z)

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

glutDisplayFunc(on_display)
glutReshapeFunc(on_reshape)
glutKeyboardFunc(on_keyboard)
glutTimerFunc(10, on_timer, 10)

glutMainLoop()
