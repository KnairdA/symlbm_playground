import numpy
from string import Template

from simulation         import Lattice, Geometry
from utility.particles  import Particles
from symbolic.generator import LBM

import symbolic.D3Q27 as D3Q27

from OpenGL.GL   import *
from OpenGL.GLUT import *

from OpenGL.GL import shaders

from pyrr import matrix44, quaternion

from geometry.sphere   import Sphere
from geometry.box      import Box
from geometry.cylinder import Cylinder

from utility.opengl import MomentsTexture

lattice_x = 256
lattice_y = 64
lattice_z = 64

updates_per_frame = 10

lid_speed = 0.05
relaxation_time = 0.51

def get_cavity_material_map(g):
    return [
        (lambda x, y, z: x > 0 and x < g.size_x-1 and
                         y > 0 and y < g.size_y-1 and
                         z > 0 and z < g.size_z-1,            1), # bulk fluid
        (lambda x, y, z: x == 1 or x == g.size_x-2 or
                         y == 1 or y == g.size_y-2 or
                         z == 1 or z == g.size_z-2,           2), # walls
        (lambda x, y, z: x == 1,                                     3), # inflow
        (lambda x, y, z: x == g.size_x-2,                     4), # outflow

        (Sphere(3*g.size_x//20, g.size_y//2, g.size_z//2, 28), 5),
        (lambda x, y, z: x > 3*g.size_x//20-30 and
                         x < 3*g.size_x//20+30 and
                         (y-g.size_y//2)*(y-g.size_y//2) + (z-g.size_z//2)*(z-g.size_z//2) < 8*8, 1),

        (lambda x, y, z: x == 0 or x == g.size_x-1 or
                         y == 0 or y == g.size_y-1 or
                         z == 0 or z == g.size_z-1,           0)  # ghost cells
    ]

boundary = Template("""
    if ( m == 2 || m == 5 ) {
        u_0 = 0.0;
        u_1 = 0.0;
        u_2 = 0.0;
    }
    if ( m == 3 ) {
        u_0 = min(time/5000.0 * $inflow, $inflow);
        u_1 = 0.0;
        u_2 = 0.0;
    }
    if ( m == 4 ) {
        rho = 1.0;
    }
""").substitute({
    "inflow": lid_speed
})

def get_projection(width, height):
    world_width = lattice_x
    world_height = world_width / width * height

    projection = matrix44.create_perspective_projection(20.0, width/height, 0.1, 1000.0)
    look = matrix44.create_look_at(
        eye    = [0, -2*lattice_x, 0],
        target = [0, 0, 0],
        up     = [0, 0, -1])

    return numpy.matmul(look, projection)

class Rotation:
    def __init__(self, shift, x = numpy.pi, z = numpy.pi):
        self.shift = shift
        self.rotation_x = x
        self.rotation_z = z
        self.update(0,0)

    def update(self, x, z):
        self.rotation_x += x
        self.rotation_z += z

        qx = quaternion.Quaternion(quaternion.create_from_eulers([self.rotation_x,0,0]))
        qz = quaternion.Quaternion(quaternion.create_from_eulers([0,0,self.rotation_z]))
        rotation = qz.cross(qx)

        self.matrix = numpy.matmul(
            matrix44.create_from_translation(self.shift),
            matrix44.create_from_quaternion(rotation)
        )
        self.inverse_matrix = numpy.linalg.inv(self.matrix)

    def get(self):
        return self.matrix

    def get_inverse(self):
        return self.inverse_matrix

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

lbm = LBM(D3Q27)

window = glut_window(fullscreen = False)

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

raycast_vertex_shader = shaders.compileShader("""
#version 430

layout (location=0) in vec4 vertex;
                   out vec3 frag_pos;

uniform mat4 projection;
uniform mat4 rotation;

void main() {
    gl_Position = projection * rotation * vertex;
    frag_pos    = vertex.xyz;
}""", GL_VERTEX_SHADER)

raycast_fragment_shader = shaders.compileShader(Template("""
#version 430

in vec3 frag_pos;

uniform mat4 inverse_rotation;

uniform sampler3D moments;

out vec4 result;

vec3 unit(vec3 v) {
    return vec3(v[0] / $size_x, v[1] / $size_y, v[2] / $size_z);
}

float norm(vec3 v) {
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

vec3 blueRedPalette(float x) {
    return mix(
        vec3(0.0, 0.0, 1.0),
        vec3(1.0, 0.0, 0.0),
        x
    );
}

void main(){
    const vec4 camera_pos = inverse_rotation * vec4(0,-2*$size_x,0,1);

    const vec3 ray = normalize(frag_pos - camera_pos.xyz);

    vec4 color = vec4(0.0,0.0,0.0,1.0);
    const float ray_length = $max_ray_length;

    for (float t = 1.0; t < ray_length; t += ray_length/80.0) {
        const vec3 sample_pos = unit(frag_pos + t*ray);
        if (norm(sample_pos) < 3.05) {
            const vec4 data = texture(moments, sample_pos);
            if (data[3] != 1.0) {
                const float norm = sqrt(data[1]*data[1]+data[2]*data[2]+data[3]*data[3]) / ($lid_speed);
                color.rgb += 0.05 * (1.0 - t/ray_length) * blueRedPalette(0.5*norm);
            } else {
                color.rgb += 0.03;
            }
        } else {
            break;
        }
    }

    result = color;
}
""").substitute({
    "size_x": lattice_x,
    "size_y": lattice_y,
    "size_z": lattice_z,
    "lid_speed": lid_speed,
    "max_ray_length": numpy.sqrt((lattice_x*lattice_x+lattice_y*lattice_y)+lattice_z*lattice_z)
}), GL_FRAGMENT_SHADER)

domain_program   = shaders.compileProgram(vertex_shader, fragment_shader)
domain_projection_id = shaders.glGetUniformLocation(domain_program, 'projection')
domain_rotation_id   = shaders.glGetUniformLocation(domain_program, 'rotation')

raycast_program = shaders.compileProgram(raycast_vertex_shader, raycast_fragment_shader)
raycast_projection_id       = shaders.glGetUniformLocation(raycast_program, 'projection')
raycast_rotation_id         = shaders.glGetUniformLocation(raycast_program, 'rotation')
raycast_inverse_rotation_id = shaders.glGetUniformLocation(raycast_program, 'inverse_rotation')

lattice = Lattice(
    descriptor   = D3Q27,
    geometry     = Geometry(lattice_x, lattice_y, lattice_z),
    moments      = lbm.moments(optimize = True),
    collide      = lbm.bgk(f_eq = lbm.equilibrium(), tau = relaxation_time),
    boundary_src = boundary,
    opengl       = True
)

moments_texture = MomentsTexture(lattice)

material_map = get_cavity_material_map(lattice.geometry)
primitives   = list(map(lambda material: material[0], filter(lambda material: not callable(material[0]), material_map)))
lattice.apply_material_map(material_map)
lattice.sync_material()

rotation = Rotation([-0.5*lattice_x, -0.5*lattice_y, -0.5*lattice_z])

cube_vertices, cube_edges = lattice.geometry.wireframe()

def on_display():
    for i in range(0,updates_per_frame):
        lattice.evolve()

    moments_texture.collect()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    shaders.glUseProgram(raycast_program)
    glUniformMatrix4fv(raycast_projection_id,       1, False, numpy.ascontiguousarray(projection))
    glUniformMatrix4fv(raycast_rotation_id,         1, False, numpy.ascontiguousarray(rotation.get()))
    glUniformMatrix4fv(raycast_inverse_rotation_id, 1, False, numpy.ascontiguousarray(rotation.get_inverse()))
    moments_texture.bind()
    Box(0,lattice.geometry.size_x,0,lattice.geometry.size_y,0,lattice.geometry.size_z).draw()

    shaders.glUseProgram(domain_program)
    glUniformMatrix4fv(domain_projection_id, 1, False, numpy.ascontiguousarray(projection))
    glUniformMatrix4fv(domain_rotation_id,   1, False, numpy.ascontiguousarray(rotation.get()))
    glLineWidth(4)
    glBegin(GL_LINES)
    for i, j in cube_edges:
        glVertex(cube_vertices[i])
        glVertex(cube_vertices[j])
    glEnd()

    glutSwapBuffers()

def on_reshape(width, height):
    global projection
    glViewport(0,0,width,height)
    projection = get_projection(width, height)

def on_keyboard(key, x, y):
    global rotation

    x = {
        b'w': -numpy.pi/20,
        b's':  numpy.pi/20
    }.get(key, 0.0)
    z = {
        b'a':  numpy.pi/20,
        b'd': -numpy.pi/20
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
