import numpy
from string import Template

from simulation         import Lattice, Geometry
from utility.particles  import Particles
from symbolic.generator import LBM

import symbolic.D3Q27 as D3Q27

from OpenGL.GL   import *
from OpenGL.GLUT import *

from OpenGL.GL import shaders

from geometry.sphere   import Sphere
from geometry.box      import Box
from geometry.cylinder import Cylinder

from utility.projection import Projection, Rotation
from utility.opengl     import MomentsTexture
from utility.mouse      import MouseDragMonitor

lattice_x = 256
lattice_y = 64
lattice_z = 64

updates_per_frame = 8

inflow = 0.05
relaxation_time = 0.515

lbm = LBM(D3Q27)

def get_cavity_material_map(g):
    return [
        (lambda x, y, z: x > 0 and x < g.size_x-1 and
                         y > 0 and y < g.size_y-1 and
                         z > 0 and z < g.size_z-1,            1), # bulk fluid
        (lambda x, y, z: x == 1 or x == g.size_x-2 or
                         y == 1 or y == g.size_y-2 or
                         z == 1 or z == g.size_z-2,           2), # walls
        (lambda x, y, z: x == 1,                              3), # inflow
        (lambda x, y, z: x == g.size_x-2,                     4), # outflow

        (Sphere(3*g.size_x//20, g.size_y//2, g.size_z//2, 29), 5),
        (lambda x, y, z: x > 3*g.size_x//20-30 and
                         x < 3*g.size_x//20+30 and
                         (y-g.size_y//2)*(y-g.size_y//2) + (z-g.size_z//2)*(z-g.size_z//2) < 8*8, 1),

        (Box(10*g.size_x//20, 11*g.size_x//20, 0, g.size_y, 1.5*g.size_z//3, g.size_z), 5),

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
    "inflow": inflow
})

def glut_window(fullscreen = False):
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE)

    if fullscreen:
        window = glutEnterGameMode()
    else:
        glutInitWindowSize(800, 500)
        glutInitWindowPosition(0, 0)
        window = glutCreateWindow("LBM")

    return window

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

    for (float t = 0.0; t < 1.0; t += 1./ray_length) {
        const vec3 sample_pos = unit(frag_pos + t*ray_length*ray);
        if (norm(sample_pos) < 3.05) {
            const vec4 data = texture(moments, sample_pos);
            if (data[3] != 1.0) {
                const float norm = sqrt(norm(data.yzw)) / $inflow;
                color.rgb += 0.01*blueRedPalette(norm);
            } else {
                color.rgb += 0.5;
                break;
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
    "inflow": inflow,
    "max_ray_length": lattice_x
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

projection = Projection(distance = 2*lattice_x)
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
    glUniformMatrix4fv(raycast_projection_id,       1, False, numpy.ascontiguousarray(projection.get()))
    glUniformMatrix4fv(raycast_rotation_id,         1, False, numpy.ascontiguousarray(rotation.get()))
    glUniformMatrix4fv(raycast_inverse_rotation_id, 1, False, numpy.ascontiguousarray(rotation.get_inverse()))
    moments_texture.bind()
    Box(0,lattice.geometry.size_x,0,lattice.geometry.size_y,0,lattice.geometry.size_z).draw()

    shaders.glUseProgram(domain_program)
    glUniformMatrix4fv(domain_projection_id, 1, False, numpy.ascontiguousarray(projection.get()))
    glUniformMatrix4fv(domain_rotation_id,   1, False, numpy.ascontiguousarray(rotation.get()))
    glLineWidth(4)
    glBegin(GL_LINES)
    for i, j in cube_edges:
        glVertex(cube_vertices[i])
        glVertex(cube_vertices[j])
    glEnd()

    glutSwapBuffers()

mouse_monitor = MouseDragMonitor(
    GLUT_LEFT_BUTTON,
    drag_callback = lambda dx, dy: rotation.update(0.005*dy, 0.005*dx),
    zoom_callback = lambda zoom: projection.update_distance(5*zoom))

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

glutDisplayFunc(on_display)
glutReshapeFunc(lambda w, h: projection.update_ratio(w, h))
glutMouseFunc(mouse_monitor.on_mouse)
glutMotionFunc(mouse_monitor.on_mouse_move)
glutTimerFunc(10, on_timer, 10)

glutMainLoop()
