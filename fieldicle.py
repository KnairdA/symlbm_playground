import pyopencl as cl
mf = cl.mem_flags
from pyopencl.tools import get_gl_sharing_context_properties

from string import Template

from OpenGL.GL import *   # OpenGL - GPU rendering interface
from OpenGL.GLU import *  # OpenGL tools (mipmaps, NURBS, perspective projection, shapes)
from OpenGL.GLUT import * # OpenGL tool to make a visualization window
from OpenGL.arrays import vbo

import numpy
import threading

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

class ParticleWindow:
    window_width  = 500
    window_height = 500

    world_width  = 20.
    world_height = 20.

    num_particles = 100000
    time_step = .005

    gtk_active = False

    def glut_window(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.window_width, self.window_height)
        glutInitWindowPosition(0, 0)
        window = glutCreateWindow("fieldicle")

        glutDisplayFunc(self.on_display)
        glutSpecialFunc(self.on_keyboard)
        glutTimerFunc(5, self.on_timer, 5)
        glutReshapeFunc(self.on_window_resize)

        glViewport(0, 0, self.window_width, self.window_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        glOrtho(
            -(self.world_width/2), self.world_width/2,
            -(self.world_height/2), self.world_height/2,
            0.1, 100.0
        )

        return(window)

    def on_keyboard(self, key, x, y):
        if key == GLUT_KEY_F1:
            self.gtk_active = True
            ParamWindow(self).show_all()

    def initial_buffers(self, num_particles):
        self.np_position = numpy.ndarray((self.num_particles, 4), dtype=numpy.float32)
        self.np_color = numpy.ndarray((num_particles, 4), dtype=numpy.float32)

        self.set_particle_start_positions()

        self.np_color[:,:] = [1.,1.,1.,1.]
        self.np_color[:,3] = numpy.random.random_sample((self.num_particles,))

        self.gl_position = vbo.VBO(data=self.np_position, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
        self.gl_position.bind()
        self.gl_color = vbo.VBO(data=self.np_color, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
        self.gl_color.bind()

        return (self.np_position, self.gl_position, self.gl_color)

    def on_timer(self, t):
        glutTimerFunc(t, self.on_timer, t)
        glutPostRedisplay()
        if self.gtk_active:
            Gtk.main_iteration_do(False)

    def set_particle_start_positions(self):
        self.np_position[:,0] = self.world_width  * numpy.random.random_sample((self.num_particles,)) - (self.world_width/2)
        self.np_position[:,1] = self.world_height * numpy.random.random_sample((self.num_particles,)) - (self.world_height/2)
        self.np_position[:,2] = 0.
        self.np_position[:,3] = 1.
        self.cl_start_position = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.np_position)

    def on_window_resize(self, width, height):
        self.window_width  = width
        self.window_height = height
        self.world_height = self.world_width / self.window_width * self.window_height;

        glViewport(0, 0, self.window_width, self.window_height)
        glLoadIdentity()
        glOrtho(
            -(self.world_width/2), self.world_width/2,
            -(self.world_height/2), self.world_height/2,
            0.1, 100.0
        )

        self.set_particle_start_positions()

    def update_field(self, fx, fy):
        self.program = cl.Program(self.context, Template(self.kernel).substitute({
            'fx': fx,
            'fy': fy,
            'time_step': self.time_step
        })).build()

    def on_display(self):
        # Update or particle positions by calling the OpenCL kernel
        cl.enqueue_acquire_gl_objects(self.queue, [self.cl_gl_position, self.cl_gl_color])
        kernelargs = (self.cl_gl_position, self.cl_gl_color, self.cl_start_position)
        self.program.update_particles(self.queue, (self.num_particles,), None, *(kernelargs))
        cl.enqueue_release_gl_objects(self.queue, [self.cl_gl_position, self.cl_gl_color])
        self.queue.finish()
        glFlush()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glTranslatef(0., 0., -1.)

        # Render the particles
        glEnable(GL_POINT_SMOOTH)
        glPointSize(1)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Set up the VBOs
        self.gl_color.bind()
        glColorPointer(4, GL_FLOAT, 0, self.gl_color)
        self.gl_position.bind()
        glVertexPointer(4, GL_FLOAT, 0, self.gl_position)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        # Draw the VBOs
        glDrawArrays(GL_POINTS, 0, self.num_particles)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        glDisable(GL_BLEND)

        glutSwapBuffers()

    def run(self):
        self.window = self.glut_window()

        self.platform = cl.get_platforms()[0]
        self.context = cl.Context(properties=[(cl.context_properties.PLATFORM, self.platform)] + get_gl_sharing_context_properties())
        self.queue = cl.CommandQueue(self.context)

        (self.np_position, self.gl_position, self.gl_color) = self.initial_buffers(self.num_particles)

        self.cl_start_position = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.np_position)

        self.cl_gl_position = cl.GLBuffer(self.context, mf.READ_WRITE, int(self.gl_position))
        self.cl_gl_color = cl.GLBuffer(self.context, mf.READ_WRITE, int(self.gl_color))

        self.kernel = """__kernel void update_particles(__global float4* position,
                                                   __global float4* color,
                                                   __global float4* start_position)
        {
            unsigned int i = get_global_id(0);
            float4 p = position[i];

            float life = color[i].w;
            life -= $time_step;

            if (life <= 0.f) {
                p = start_position[i];
                life = 1.0f;
            }

            p.x += ($fx) * $time_step;
            p.y += ($fy) * $time_step;

            position[i] = p;
            color[i].w = life;
        }"""
        self.program = cl.Program(self.context, Template(self.kernel).substitute({
            'fx': 'cos(p.x)',
            'fy': 'sin(p.y*p.x)',
            'time_step': self.time_step
        })).build()

        glutMainLoop()


particleWindow = ParticleWindow()

glfwThread = threading.Thread(target=particleWindow.run)
glfwThread.start()

class ParamWindow(Gtk.Dialog):
    def __init__(self, particleWin):
        Gtk.Dialog.__init__(self, title="Field Parameters")
        self.particleWin = particleWin

        self.updateBtn = Gtk.Button(label="Update field")
        self.updateBtn.connect("clicked", self.on_update_clicked)

        self.entryFx = Gtk.Entry()
        self.entryFx.set_text("cos(p.x)")
        self.entryFy = Gtk.Entry()
        self.entryFy.set_text("sin(p.y*p.x)")

        layout = self.get_content_area()

        layout.add(self.entryFx)
        layout.add(self.entryFy)
        layout.add(self.updateBtn)

    def on_update_clicked(self, widget):
        self.particleWin.update_field(
            self.entryFx.get_text(),
            self.entryFy.get_text()
        )
