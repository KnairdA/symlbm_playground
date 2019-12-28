import numpy

from mako.template import Template
from mako.lookup import TemplateLookup

from pathlib import Path

from simulation         import Lattice, Geometry
from utility.particles  import Particles
from symbolic.generator import LBM

import symbolic.D3Q19 as D3Q19

from OpenGL.GL   import *
from OpenGL.GLUT import *

from OpenGL.GL import shaders

from geometry.box import Box

from utility.projection import Projection, Rotation
from utility.opengl     import MomentsTexture
from utility.mouse      import MouseDragMonitor, MouseScrollMonitor

lattice_x = 180
lattice_y = 100
lattice_z = 100

updates_per_frame = 5

inflow = 0.075
relaxation_time = 0.51

lbm = LBM(D3Q19)

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

grid_fin = """
v = rotate_z(translate(v, v3(center.x/2, center.y, center.z)), -0.8);
float width = 1;
float angle = 0.64;

return add(
    sadd(
        sub(
            rounded(box(v, v3(5, 28, 38)), 1),
            rounded(box(v, v3(6, 26, 36)), 1)
        ),
        cylinder(translate(v, v3(0,0,-45)), 5, 12),
        1
    ),
    sintersect(
        box(v, v3(5, 28, 38)),
        add(
            add(
                box(rotate_x(v, angle), v3(10, width, 100)),
                box(rotate_x(v, -angle), v3(10, width, 100))
            ),
            add(
                add(
                    add(
                        box(rotate_x(translate(v, v3(0,0,25)), angle), v3(10, width, 100)),
                        box(rotate_x(translate(v, v3(0,0,25)), -angle), v3(10, width, 100))
                    ),
                    add(
                        box(rotate_x(translate(v, v3(0,0,-25)), angle), v3(10, width, 100)),
                        box(rotate_x(translate(v, v3(0,0,-25)), -angle), v3(10, width, 100))
                    )
                ),
                add(
                    add(
                        box(rotate_x(translate(v, v3(0,0,50)), angle), v3(10, width, 100)),
                        box(rotate_x(translate(v, v3(0,0,50)), -angle), v3(10, width, 100))
                    ),
                    add(
                        box(rotate_x(translate(v, v3(0,0,-50)), angle), v3(10, width, 100)),
                        box(rotate_x(translate(v, v3(0,0,-50)), -angle), v3(10, width, 100))
                    )
                )
            )
        ),
        2
    )
);
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

in vec3 frag_pos;

uniform vec4 camera_pos;

uniform sampler3D moments;

out vec4 result;

vec3 unit(vec3 v) {
    return vec3(v[0] / ${size_x}, v[1] / ${size_y}, v[2] / ${size_z});
}

const vec3 center = vec3(${size_x/2.5}, ${size_y/2}, ${size_z/2});

#define EPSILON 1e-1
#define RAYMARCH_STEPS 64
#define OBSTACLE_STEPS 16

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

float sdf(vec3 v) {
    ${sdf_source}
}

vec3 sdf_normal(vec3 v) {
    return normalize(vec3(
        sdf(vec3(v.x + EPSILON, v.y, v.z)) - sdf(vec3(v.x - EPSILON, v.y, v.z)),
        sdf(vec3(v.x, v.y + EPSILON, v.z)) - sdf(vec3(v.x, v.y - EPSILON, v.z)),
        sdf(vec3(v.x, v.y, v.z + EPSILON)) - sdf(vec3(v.x, v.y, v.z - EPSILON))
    ));
}

vec3 palette(float x) {
    return mix(
        vec3(0.0, 0.0, 0.0),
        vec3(1.0, 0.0, 0.0),
        x
    );
}

float distanceToLattice(vec3 v) {
  return box(v, vec3(${size_x},${size_y},${size_z}));
}

float maxRayLength(vec3 origin, vec3 ray) {
    return max(1.0, ${max_ray_length} - distanceToLattice(origin + ${max_ray_length}*ray));
}

vec4 trace_obstacle(vec3 origin, vec3 ray, float delta) {
    vec3 color      = vec3(0);
    vec3 sample_pos = origin;
    float ray_dist = 0.0;

    for (int i = 0; i < OBSTACLE_STEPS; ++i) {
        const float sdf_dist = sdf(sample_pos);
        ray_dist += sdf_dist;

        if (ray_dist > delta) {
            return vec4(0.0);
        }

        if (abs(sdf_dist) < EPSILON) {
            const vec3 n = normalize(sdf_normal(sample_pos));
            return vec4(color + abs(dot(n, ray)), 1.0);
        } else {
            sample_pos = origin + ray_dist*ray;
        }
    }

    return vec4(0.0);
}

vec3 trace(vec3 pos, vec3 ray) {
    const float delta = maxRayLength(pos, ray) / RAYMARCH_STEPS;

    vec3 color = vec3(0.0);

    for (int i=0; i < RAYMARCH_STEPS; ++i) {
        const vec3 sample_pos = pos + i*delta*ray;
        const vec4 data = texture(moments, unit(sample_pos));
        if (sdf(sample_pos) > delta) {
            color += 0.5/RAYMARCH_STEPS * palette(length(data.yzw) / ${inflow});
        } else {
            const vec4 obstacle_color = trace_obstacle(sample_pos, ray, delta);
            if (obstacle_color.w == 1.0) {
                return color + obstacle_color.xyz;
            } else {
                color += 0.5/RAYMARCH_STEPS * palette(length(data.yzw) / ${inflow});
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
    sdf_source = grid_fin
), GL_FRAGMENT_SHADER)

domain_program   = shaders.compileProgram(vertex_shader, fragment_shader)
domain_projection_id = shaders.glGetUniformLocation(domain_program, 'projection')
domain_rotation_id   = shaders.glGetUniformLocation(domain_program, 'rotation')

raycast_program = shaders.compileProgram(raycast_vertex_shader, raycast_fragment_shader)
raycast_projection_id       = shaders.glGetUniformLocation(raycast_program, 'projection')
raycast_rotation_id         = shaders.glGetUniformLocation(raycast_program, 'rotation')
raycast_camera_pos_id = shaders.glGetUniformLocation(raycast_program, 'camera_pos')

lattice = Lattice(
    descriptor   = D3Q19,
    geometry     = Geometry(lattice_x, lattice_y, lattice_z),
    moments      = lbm.moments(optimize = True),
    collide      = lbm.bgk(f_eq = lbm.equilibrium(), tau = relaxation_time, optimize = True),
    boundary_src = boundary,
    opengl       = True
)

lattice.setup_channel_with_sdf_obstacle(grid_fin)

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
