import numpy
from string import Template

from simulation         import Lattice, Geometry
from utility.particles  import Particles
from symbolic.generator import LBM

import symbolic.D3Q19 as D3Q19

from OpenGL.GL   import *
from OpenGL.GLUT import *

from OpenGL.GL import shaders

from utility.projection import Projection, Rotation
from utility.opengl     import MomentsVertexBuffer
from utility.mouse      import MouseDragMonitor, MouseScrollMonitor

lattice_x = 64
lattice_y = 96
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
particle_projection_id = shaders.glGetUniformLocation(particle_program, 'projection')
particle_rotation_id   = shaders.glGetUniformLocation(particle_program, 'rotation')

geometry_program = shaders.compileProgram(vertex_shader, fragment_shader)
geometry_projection_id = shaders.glGetUniformLocation(geometry_program, 'projection')
geometry_rotation_id   = shaders.glGetUniformLocation(geometry_program, 'rotation')

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
    lattice,
    numpy.mgrid[
        8*lattice.geometry.size_x//10:9*lattice.geometry.size_x//10:10j,
        lattice.geometry.size_y//10:9*lattice.geometry.size_y//10:particle_count/100j,
        8*lattice.geometry.size_z//10:9*lattice.geometry.size_z//10:10j,
    ].reshape(3,-1).T)

projection = Projection(distance = 2*lattice_x)
rotation = Rotation([-lattice_x/2, -lattice_y/2, -lattice_z/2])

cube_vertices, cube_edges = lattice.geometry.wireframe()

def on_display():
    for i in range(0,updates_per_frame):
        lattice.evolve()

    lattice.update_moments()

    for i in range(0,updates_per_frame):
        particles.update(aging = True)

    lattice.sync()

    glClear(GL_COLOR_BUFFER_BIT)

    shaders.glUseProgram(particle_program)
    glUniformMatrix4fv(particle_projection_id, 1, False, numpy.ascontiguousarray(projection.get()))
    glUniformMatrix4fv(particle_rotation_id,   1, False, numpy.ascontiguousarray(rotation.get()))
    particles.bind()
    glEnable(GL_POINT_SMOOTH)
    glPointSize(1)
    glDrawArrays(GL_POINTS, 0, particles.count)

    shaders.glUseProgram(geometry_program)
    glUniformMatrix4fv(geometry_projection_id, 1, False, numpy.ascontiguousarray(projection.get()))
    glUniformMatrix4fv(geometry_rotation_id,   1, False, numpy.ascontiguousarray(rotation.get()))
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glLineWidth(2)
    glBegin(GL_LINES)
    for i, j in cube_edges:
        glVertex3fv(cube_vertices[i])
        glVertex3fv(cube_vertices[j])
    glEnd()

    glutSwapBuffers()

mouse_monitors = [
    MouseDragMonitor(GLUT_LEFT_BUTTON, lambda dx, dy: rotation.update(0.005*dy, 0.005*dx)),
    MouseScrollMonitor(lambda zoom: projection.update_distance(5*zoom))
]

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

glutDisplayFunc(on_display)
glutReshapeFunc(lambda w, h: projection.update_ratio(w, h))
glutMouseFunc(lambda *args: list(map(lambda m: m.on_mouse(*args), mouse_monitors)))
glutMotionFunc(lambda *args: list(map(lambda m: m.on_mouse_move(*args), mouse_monitors)))
glutTimerFunc(10, on_timer, 10)

glutMainLoop()
