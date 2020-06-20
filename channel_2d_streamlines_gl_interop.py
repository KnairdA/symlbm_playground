import numpy
from string import Template

from simulation          import Lattice, Geometry
from utility.streamline  import Streamlines
from symbolic.generator  import LBM

import symbolic.D2Q9 as D2Q9

from OpenGL.GL   import *
from OpenGL.GLUT import *

from OpenGL.GL import shaders

from pyrr import matrix44

lattice_x = 1024
lattice_y = 256

updates_per_frame = 10

inflow = 0.01
relaxation_time = 0.51

def circle(cx, cy, r):
    return lambda x, y: (x - cx)**2 + (y - cy)**2 < r*r

def get_channel_material_map(geometry):
    return [
        (lambda x, y: x > 0 and x < geometry.size_x-1 and y > 0 and y < geometry.size_y-1, 1), # bulk fluid

        (lambda x, y: x == 1,                 3), # inflow
        (lambda x, y: x == geometry.size_x-2, 4), # outflow
        (lambda x, y: y == 1,                 2), # bottom
        (lambda x, y: y == geometry.size_y-2, 2), # top

        (circle(1.0*geometry.size_x//6, 1*geometry.size_y//3, geometry.size_y//5), 2),
        (circle(1.5*geometry.size_x//6, 2*geometry.size_y//3, geometry.size_y//6), 2),

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

def get_projection(width, height):
    world_width = lattice_x
    world_height = world_width / width * height

    projection  = matrix44.create_orthogonal_projection(-world_width/2, world_width/2, -world_height/2, world_height/2, -1, 1)
    translation = matrix44.create_from_translation([-lattice_x/2, -lattice_y/2, 0])

    point_size = width / world_width

    return numpy.matmul(translation, projection), point_size

def glut_window(fullscreen = False):
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    if fullscreen:
        window = glutEnterGameMode()
    else:
        glutInitWindowSize(800, 600)
        glutInitWindowPosition(0, 0)
        window = glutCreateWindow("LBM")

    return window

lbm = LBM(D2Q9)

window = glut_window(fullscreen = False)

vertex_shader = shaders.compileShader("""
#version 430

layout (location=0) in vec4 vertex;
                   out vec2 frag_pos;

uniform mat4 projection;

void main() {
    gl_Position = projection * vertex;
    frag_pos    = vertex.xy;
}""", GL_VERTEX_SHADER)

fragment_shader = shaders.compileShader(Template("""
#version 430

in vec2 frag_pos;

uniform sampler2D moments;

out vec4 result;

vec2 unit(vec2 v) {
    return vec2(v[0] / $size_x, v[1] / $size_y);
}

void main(){
    const vec2 sample_pos = unit(frag_pos);
    result = texture(moments, sample_pos);
}
""").substitute({
    "size_x": lattice_x,
    "size_y": lattice_y,
    "inflow": inflow
}), GL_FRAGMENT_SHADER)

shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
projection_id = shaders.glGetUniformLocation(shader_program, 'projection')

lattice = Lattice(
    descriptor   = D2Q9,
    geometry     = Geometry(lattice_x, lattice_y),
    moments      = lbm.moments(optimize = False),
    collide      = lbm.bgk(f_eq = lbm.equilibrium(), tau = relaxation_time),
    boundary_src = boundary,
    opengl       = True
)

lattice.apply_material_map(
    get_channel_material_map(lattice.geometry))
lattice.sync_material()

streamline_texture = Streamlines(
    lattice,
    list(map(lambda y: [2, y*lattice.geometry.size_y//48], range(1,48))))

def on_display():
    for i in range(0,updates_per_frame):
        lattice.evolve()

    lattice.update_moments()
    streamline_texture.update()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    shaders.glUseProgram(shader_program)
    glUniformMatrix4fv(projection_id, 1, False, numpy.asfortranarray(projection))
    streamline_texture.bind()

    glBegin(GL_POLYGON)
    glVertex(0,0,0)
    glVertex(lattice.geometry.size_x,0,0)
    glVertex(lattice.geometry.size_x,lattice.geometry.size_y,0)
    glVertex(0,lattice.geometry.size_y,0)
    glEnd()

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
