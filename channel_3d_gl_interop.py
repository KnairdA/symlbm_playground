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

lattice_x = 256
lattice_y = 64
lattice_z = 64

cylinder_r = 10

updates_per_frame = 10
particle_count = 50000

lid_speed = 0.05
relaxation_time = 0.51

def circle(cx, cy, cz, r):
    return lambda x, y, z: (x - cx)**2 + (y - cy)**2 + (z - cz)**2 < r*r

def cylinder(cx, cz, r):
    return lambda x, y, z: (x - cx)**2 + (z - cz)**2 < r*r

def get_cavity_material_map(geometry):
    return [
        (lambda x, y, z: x > 0 and x < geometry.size_x-1 and
                         y > 0 and y < geometry.size_y-1 and
                         z > 0 and z < geometry.size_z-1,            1), # bulk fluid
        (lambda x, y, z: x == 1 or x == geometry.size_x-2 or
                         y == 1 or y == geometry.size_y-2 or
                         z == 1 or z == geometry.size_z-2,           2), # walls
        (lambda x, y, z: x == 1,                                     3), # inflow
        (lambda x, y, z: x == geometry.size_x-2,                     4), # outflow
        (cylinder(geometry.size_x//6, geometry.size_z//2, cylinder_r), 5), # obstacle
        (lambda x, y, z: x == 0 or x == geometry.size_x-1 or
                         y == 0 or y == geometry.size_y-1 or
                         z == 0 or z == geometry.size_z-1,           0)  # ghost cells
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

    point_size = 1

    return numpy.matmul(look, projection), point_size

class Rotation:
    def __init__(self, shift, x = numpy.pi, z = numpy.pi):
        self.shift = shift
        self.rotation_x = x
        self.rotation_z = z

    def update(self, x, z):
        self.rotation_x += x
        self.rotation_z += z

    def get(self):
        qx = quaternion.Quaternion(quaternion.create_from_eulers([self.rotation_x,0,0]))
        qz = quaternion.Quaternion(quaternion.create_from_eulers([0,0,self.rotation_z]))
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

lbm = LBM(D3Q27)

window = glut_window(fullscreen = False)

particle_shader = shaders.compileShader("""
#version 430

layout (location=0) in vec4 particles;

out VS_OUT {
    vec3 color;
} vs_out;

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

    vs_out.color = fire(1.0-particles[3]);
}""", GL_VERTEX_SHADER)

geometry_shader = shaders.compileShader("""
#version 430

layout (points) in;
layout (triangle_strip, max_vertices=4) out;

in VS_OUT {
    vec3 color;
} gs_in[];

out vec3 color;

void emitSquareAt(vec4 position) {
    const float size = 0.5;

    gl_Position = position + vec4(-size, -size, 0.0, 0.0);
    EmitVertex();
    gl_Position = position + vec4( size, -size, 0.0, 0.0);
    EmitVertex();
    gl_Position = position + vec4(-size,  size, 0.0, 0.0);
    EmitVertex();
    gl_Position = position + vec4( size,  size, 0.0, 0.0);
    EmitVertex();
}

void main() {
    color = gs_in[0].color;
    emitSquareAt(gl_in[0].gl_Position);
    EndPrimitive();
}""", GL_GEOMETRY_SHADER)

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

lighting_vertex_shader = shaders.compileShader("""
#version 430

layout (location=0) in vec4 vertex;
layout (location=2) in vec4 normal;
                   out vec3 color;
                   out vec3 frag_pos;
                   out vec3 frag_normal;

uniform mat4 projection;
uniform mat4 rotation;

void main() {
    gl_Position = projection * rotation * vertex;
    frag_pos    = vertex.xyz;
    frag_normal = normalize(normal.xyz);
    color = vec3(0.6,0.6,0.6);
}""", GL_VERTEX_SHADER)

fragment_shader = shaders.compileShader("""
#version 430

in vec3 color;

void main(){
    gl_FragColor = vec4(color.xyz, 0.0);
}""", GL_FRAGMENT_SHADER)

lighting_fragment_shader = shaders.compileShader(Template("""
#version 430

in vec3 color;
in vec3 frag_pos;
in vec3 frag_normal;

uniform mat4 projection;
uniform mat4 rotation;

out vec4 result;

void main(){
    const vec4 light_pos = rotation * vec4($size_x/2,-$size_x,$size_z/2,1);
    const vec3 light_color = vec3(1.0,1.0,1.0);

    const vec3 ray = light_pos.xyz - frag_pos;
    float brightness = dot(frag_normal, ray) / length(ray);
    brightness = clamp(brightness, 0, 1);

    result = vec4(max(0.4,brightness) * light_color * color.rgb, 1.0);
}
""").substitute({
    "size_x": lattice_x,
    "size_y": lattice_y,
    "size_z": lattice_z
}), GL_FRAGMENT_SHADER)

particle_program = shaders.compileProgram(particle_shader, geometry_shader, fragment_shader)
projection_id = shaders.glGetUniformLocation(particle_program, 'projection')
rotation_id   = shaders.glGetUniformLocation(particle_program, 'rotation')

domain_program   = shaders.compileProgram(vertex_shader, fragment_shader)
obstacle_program = shaders.compileProgram(lighting_vertex_shader, lighting_fragment_shader)

lattice = Lattice(
    descriptor   = D3Q27,
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
        2*lattice.geometry.size_x//100:4*lattice.geometry.size_x//100:particle_count/10000j,
        lattice.geometry.size_y//10:9*lattice.geometry.size_y//10:100j,
        lattice.geometry.size_z//10:9*lattice.geometry.size_z//10:100j,
    ].reshape(3,-1).T)

rotation = Rotation([-lattice_x/2, -lattice_y/2, -lattice_z/2])

cube_vertices, cube_edges = lattice.geometry.wireframe()

def draw_cylinder(cx, cz, height, radius, num_slices):
    r = radius
    h = height
    n = float(num_slices)

    circle_pts = []
    for i in range(int(n) + 1):
        angle = 2 * numpy.pi * (i/n)
        x = cx + r * numpy.cos(angle)
        z = cz + r * numpy.sin(angle)
        pt = (x, z)
        circle_pts.append(pt)

    glBegin(GL_TRIANGLE_FAN)
    glNormal(0,-1, 0)
    glVertex(cx, 0, cz)
    for (x, z) in circle_pts:
        y = 0
        glNormal(0,-1, 0)
        glVertex(x, y, z)
    glEnd()

    glBegin(GL_TRIANGLE_FAN)
    glNormal(0, 1, 0)
    glVertex(cx, h, cz)
    for (x, z) in circle_pts:
        y = h
        glNormal(0, 1, 0)
        glVertex(x, y, z)
    glEnd()

    glBegin(GL_TRIANGLE_STRIP)
    for (x, z) in circle_pts:
        y = h
        glNormal(x-cx, 0, z-cz)
        glVertex(x, 0, z)
        glNormal(x-cx, 0, z-cz)
        glVertex(x, h, z)
    glEnd()

def on_display():
    for i in range(0,updates_per_frame):
        lattice.evolve()

    lattice.collect_gl_moments()

    for i in range(0,updates_per_frame):
        lattice.update_gl_particles(particles, aging = True)

    lattice.sync()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    glEnableClientState(GL_VERTEX_ARRAY)
    particles.gl_particles.bind()

    shaders.glUseProgram(particle_program)
    glUniformMatrix4fv(projection_id, 1, False, numpy.ascontiguousarray(projection))
    glUniformMatrix4fv(rotation_id, 1, False, numpy.ascontiguousarray(rotation.get()))
    glVertexPointer(4, GL_FLOAT, 0, particles.gl_particles)
    glEnable(GL_POINT_SMOOTH)
    glPointSize(point_size)
    glDrawArrays(GL_POINTS, 0, particles.count)

    shaders.glUseProgram(domain_program)
    glUniformMatrix4fv(projection_id, 1, False, numpy.ascontiguousarray(projection))
    glUniformMatrix4fv(rotation_id, 1, False, numpy.ascontiguousarray(rotation.get()))
    glLineWidth(point_size)
    glBegin(GL_LINES)
    for i, j in cube_edges:
        glVertex(cube_vertices[i])
        glVertex(cube_vertices[j])
    glEnd()

    shaders.glUseProgram(obstacle_program)
    glUniformMatrix4fv(projection_id, 1, False, numpy.ascontiguousarray(projection))
    glUniformMatrix4fv(rotation_id, 1, False, numpy.ascontiguousarray(rotation.get()))
    draw_cylinder(lattice.geometry.size_x//6, lattice.geometry.size_z//2, lattice.geometry.size_y, cylinder_r, 32)

    glutSwapBuffers()

def on_reshape(width, height):
    global projection, point_size
    glViewport(0,0,width,height)
    projection, point_size = get_projection(width, height)

def on_keyboard(key, x, y):
    global rotation

    x = {
        b'w': -numpy.pi/10,
        b's':  numpy.pi/10
    }.get(key, 0.0)
    z = {
        b'a':  numpy.pi/10,
        b'd': -numpy.pi/10
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
