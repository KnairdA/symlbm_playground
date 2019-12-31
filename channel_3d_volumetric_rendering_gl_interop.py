import numpy

from mako.template import Template
from mako.lookup import TemplateLookup

from pathlib import Path

from simulation         import Lattice, Geometry
from utility.particles  import Particles
from symbolic.generator import LBM

import symbolic.D3Q27 as D3Q27

from OpenGL.GL   import *
from OpenGL.GLUT import *

from OpenGL.GL import shaders

from geometry.box import Box

from utility.projection import Projection, Rotation
from utility.opengl     import MomentsTexture
from utility.mouse      import MouseDragMonitor, MouseScrollMonitor

lattice_x = 200
lattice_y = 64
lattice_z = 64

updates_per_frame = 6

inflow = 0.01
relaxation_time = 0.51

lbm = LBM(D3Q27)

boundary = Template("""
    if ( m == 2 ) {
        u_0 = 0.0;
        u_1 = 0.0;
        u_2 = 0.0;
    }
    if ( m == 3 ) {
        u_0 = min(time/5000.0 * ${inflow}, ${inflow});
        u_1 = 0.0;
        u_2 = 0.0;
    }
    if ( m == 4 ) {
        rho = 1.0;
    }
""").render(
    inflow = inflow
)

channel = """
float sdf(vec3 v) {
    return add(
        add(
            ssub(
                box(translate(v, v3(25,center.y,center.z)), v3(5,32,32)),
                add(
                    sphere(translate(v, v3(20,0.5*center.y,1.5*center.z)), 10),
                    sphere(translate(v, v3(30,1.5*center.y,0.5*center.z)), 10)
                ),
                2
            ),
            ssub(
                box(translate(v, v3(85,center.y,center.z)), v3(5,32,32)),
                add(
                    sphere(translate(v, v3(90,1.5*center.y,1.5*center.z)), 10),
                    sphere(translate(v, v3(80,0.5*center.y,0.5*center.z)), 10)
                ),
                2
            )
        ),
        ssub(
            box(translate(v, v3(145,center.y,center.z)), v3(5,32,32)),
            add(
                cylinder(rotate_y(translate(v, v3(145,1.5*center.y,0.5*center.z)), 1), 10, 10),
                cylinder(rotate_y(translate(v, v3(145,0.5*center.y,1.5*center.z)), -1), 10, 10)
            ),
            2
        )
    );
}

float sdf_bounding(vec3 v) {
    return add(
        add(
            box(translate(v, v3(25,center.y,center.z)), v3(5,32,32)),
            box(translate(v, v3(85,center.y,center.z)), v3(5,32,32))
        ),
        box(translate(v, v3(145,center.y,center.z)), v3(5,32,32))
    );
}
"""

def glut_window(fullscreen = False):
    glutInit(sys.argv)
    glutSetOption(GLUT_MULTISAMPLE, 8)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_MULTISAMPLE)

    if fullscreen:
        window = glutEnterGameMode()
    else:
        glutInitWindowSize(800, 500)
        glutInitWindowPosition(0, 0)
        window = glutCreateWindow("LBM")

    return window

window = glut_window(fullscreen = False)

vertex_shader = shaders.compileShader("""
#version 430

layout (location=0) in vec4 vertex;

uniform mat4 projection;
uniform mat4 rotation;

void main() {
    gl_Position = projection * rotation * vertex;
}""", GL_VERTEX_SHADER)

fragment_shader = shaders.compileShader("""
#version 430

in vec3 color;

void main(){
    gl_FragColor = vec4(vec3(0.5), 1.0);
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

#define EPSILON 1e-1
#define RAYMARCH_STEPS 64
#define OBSTACLE_STEPS 16

in vec3 frag_pos;

uniform vec4 camera_pos;

uniform sampler3D moments;

out vec4 result;

const vec3 cuboid = vec3(${size_x}, ${size_y}, ${size_z});
const vec3 center = vec3(${size_x/2.5}, ${size_y/2}, ${size_z/2});

vec3 v3(float x, float y, float z) {
    return vec3(x,y,z);
}

vec2 v2(float x, float y) {
    return vec2(x,y);
}

vec3 fabs(vec3 x) {
    return abs(x);
}

float fabs(float x) {
    return abs(x);
}

<%include file="template/sdf.lib.glsl.mako"/>

${sdf_source}

vec3 sdf_normal(vec3 v) {
    return normalize(vec3(
        sdf(vec3(v.x + EPSILON, v.y, v.z)) - sdf(vec3(v.x - EPSILON, v.y, v.z)),
        sdf(vec3(v.x, v.y + EPSILON, v.z)) - sdf(vec3(v.x, v.y - EPSILON, v.z)),
        sdf(vec3(v.x, v.y, v.z + EPSILON)) - sdf(vec3(v.x, v.y, v.z - EPSILON))
    ));
}

vec3 unit(vec3 v) {
    return v / cuboid;
}

vec3 palette(float x) {
    return mix(
        vec3(0.251, 0.498, 0.498),
        vec3(0.502, 0.082, 0.082),
        x
    );
}

vec3 getVelocityColorAt(vec3 pos) {
    const vec4 data = texture(moments, unit(pos));
    return palette(length(data.yzw) / ${2*inflow});
}

float distanceToLattice(vec3 v) {
  return box(v, vec3(${size_x},${size_y},${size_z}));
}

float maxRayLength(vec3 origin, vec3 ray) {
    return max(1.0, ${max_ray_length} - distanceToLattice(origin + ${max_ray_length}*ray));
}

vec4 trace_obstacle(vec3 origin, vec3 ray, float delta) {
    vec3 sample_pos = origin;
    float ray_dist = 0.0;

    for (int i = 0; i < OBSTACLE_STEPS; ++i) {
        const float sdf_dist = sdf(sample_pos);
        ray_dist += sdf_dist;

        if (ray_dist > delta) {
            return vec4(0.0);
        }

        if (abs(sdf_dist) < EPSILON) {
            const vec3 color = vec3(0.5);
            const vec3 n = normalize(sdf_normal(sample_pos));
            return vec4(abs(dot(n, ray)) * color, 1.0);
        } else {
            sample_pos = origin + ray_dist*ray;
        }
    }

    return vec4(0.0);
}

vec3 trace(vec3 pos, vec3 ray) {
    const float depth = maxRayLength(pos, ray);
    const float delta = depth / RAYMARCH_STEPS;
    const float gamma = 0.5 / RAYMARCH_STEPS;

    vec3 color = vec3(0.0);
    vec3 sample_pos = pos;

    for (int i=0; i < RAYMARCH_STEPS; ++i) {
        sample_pos += delta*ray;

        if (sdf_bounding(sample_pos) > delta) {
            color += gamma * getVelocityColorAt(sample_pos);
        } else {
            const vec4 obstacle_color = trace_obstacle(sample_pos, ray, delta);
            if (obstacle_color.w == 1.0) {
                return color + obstacle_color.xyz;
            } else {
                color += gamma * getVelocityColorAt(sample_pos);
            }
        }
    }

    return color;
}

void main(){
    const vec3 ray = normalize(frag_pos - camera_pos.xyz);

    result = vec4(trace(frag_pos, ray), 1.0);
}
""", lookup = TemplateLookup(directories = [
    Path(__file__).parent
])).render(
    size_x = lattice_x,
    size_y = lattice_y,
    size_z = lattice_z,
    inflow = inflow,
    max_ray_length = max(lattice_x,lattice_y,lattice_z)**3,
    sdf_source = channel
), GL_FRAGMENT_SHADER)

domain_program   = shaders.compileProgram(vertex_shader, fragment_shader)
domain_projection_id = shaders.glGetUniformLocation(domain_program, 'projection')
domain_rotation_id   = shaders.glGetUniformLocation(domain_program, 'rotation')

raycast_program = shaders.compileProgram(raycast_vertex_shader, raycast_fragment_shader)
raycast_projection_id       = shaders.glGetUniformLocation(raycast_program, 'projection')
raycast_rotation_id         = shaders.glGetUniformLocation(raycast_program, 'rotation')
raycast_camera_pos_id = shaders.glGetUniformLocation(raycast_program, 'camera_pos')

lattice = Lattice(
    descriptor   = D3Q27,
    geometry     = Geometry(lattice_x, lattice_y, lattice_z),
    moments      = lbm.moments(optimize = True),
    collide      = lbm.bgk(f_eq = lbm.equilibrium(), tau = relaxation_time, optimize = True),
    boundary_src = boundary,
    opengl       = True
)

lattice.setup_channel_with_sdf_obstacle(channel)

moments_texture = MomentsTexture(lattice)

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

    camera_pos = numpy.matmul([0,-projection.distance,0,1], rotation.get_inverse())

    shaders.glUseProgram(domain_program)
    glUniformMatrix4fv(domain_projection_id, 1, False, numpy.ascontiguousarray(projection.get()))
    glUniformMatrix4fv(domain_rotation_id,   1, False, numpy.ascontiguousarray(rotation.get()))
    glLineWidth(3)
    glBegin(GL_LINES)
    for i, j in cube_edges:
        glVertex(cube_vertices[i])
        glVertex(cube_vertices[j])
    glEnd()

    shaders.glUseProgram(raycast_program)
    glUniformMatrix4fv(raycast_projection_id, 1, False, numpy.ascontiguousarray(projection.get()))
    glUniformMatrix4fv(raycast_rotation_id,   1, False, numpy.ascontiguousarray(rotation.get()))
    glUniform4fv(raycast_camera_pos_id, 1, camera_pos)
    moments_texture.bind()
    Box(0,lattice.geometry.size_x,0,lattice.geometry.size_y,0,lattice.geometry.size_z).draw()

    glutSwapBuffers()

mouse_monitors = [
    MouseDragMonitor(GLUT_LEFT_BUTTON,  lambda dx, dy: rotation.update(0.005*dy, 0.005*dx)),
    MouseDragMonitor(GLUT_RIGHT_BUTTON, lambda dx, dy: rotation.shift(0.25*dx, 0.25*dy)),
    MouseScrollMonitor(lambda zoom: projection.update_distance(5*zoom))
]

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

glutDisplayFunc(on_display)
glutReshapeFunc(lambda w, h: projection.update_ratio(w, h))
glutMouseFunc(lambda *args: list(map(lambda m: m.on_mouse(*args), mouse_monitors)))
glutMotionFunc(lambda *args: list(map(lambda m: m.on_mouse_move(*args), mouse_monitors)))
glutTimerFunc(30, on_timer, 30)

glutMainLoop()
