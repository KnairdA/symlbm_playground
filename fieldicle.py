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
    width = 800
    height = 600
    num_particles = 100000
    time_step = .005
    mouse_old = {'x': 0., 'y': 0.}
    rotate = {'x': 0., 'y': 0., 'z': 0.}
    translate = {'x': 0., 'y': 0., 'z': 0.}
    initial_translate = {'x': 0., 'y': 0., 'z': -10}

    def glut_window(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        window = glutCreateWindow("fieldicle")

        glutDisplayFunc(self.on_display)
        glutMouseFunc(self.on_click)
        glutMotionFunc(self.on_mouse_move)
        glutTimerFunc(10, self.on_timer, 10)

        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60., self.width / float(self.height), .1, 1000.)

        return(window)

    def initial_buffers(self, num_particles):
        self.np_position = numpy.ndarray((self.num_particles, 4), dtype=numpy.float32)
        self.np_color = numpy.ndarray((num_particles, 4), dtype=numpy.float32)

        self.np_position[:,0] = 10*numpy.random.random_sample((self.num_particles,)) - 5
        self.np_position[:,1] = 10*numpy.random.random_sample((self.num_particles,)) - 5
        self.np_position[:,2] = 0.
        self.np_position[:,3] = 1.

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

    def on_click(self, button, state, x, y):
        self.mouse_old['x'] = x
        self.mouse_old['y'] = y

    def updateField(self, fx, fy, fz):
        self.program = cl.Program(self.context, Template(self.kernel).substitute({
            'fx': fx,
            'fy': fy,
            'fz': fz,
            'time_step': self.time_step
        })).build()

    def on_mouse_move(self, x, y):
        self.rotate['x'] += (y - self.mouse_old['y']) * .2
        self.rotate['y'] += (x - self.mouse_old['x']) * .2

        self.mouse_old['x'] = x
        self.mouse_old['y'] = y

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

        # Handle mouse transformations
        glTranslatef(self.initial_translate['x'], self.initial_translate['y'], self.initial_translate['z'])
        glRotatef(self.rotate['x'], 1, 0, 0)
        glRotatef(self.rotate['y'], 0, 1, 0)
        glTranslatef(self.translate['x'], self.translate['y'], self.translate['z'])

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

        (self.np_position, self.gl_position, self.gl_color) = self.initial_buffers(self.num_particles)

        self.platform = cl.get_platforms()[0]
        self.context = cl.Context(properties=[(cl.context_properties.PLATFORM, self.platform)] + get_gl_sharing_context_properties())
        self.queue = cl.CommandQueue(self.context)

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

            p.x += $fx * $time_step;
            p.y += $fy * $time_step;
            p.z += $fz * $time_step;

            position[i] = p;
            color[i].w = life;
        }"""
        self.program = cl.Program(self.context, Template(self.kernel).substitute({
            'fx': 'cos(p.x)',
            'fy': 'sin(p.y*p.x)',
            'fz': '0',
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

        self.button = Gtk.Button(label="Update field")
        self.button.connect("clicked", self.on_button_clicked)

        self.entryFx = Gtk.Entry()
        self.entryFy = Gtk.Entry()
        self.entryFz = Gtk.Entry()

        self.get_content_area().add(self.button)
        self.get_content_area().add(self.entryFx)
        self.get_content_area().add(self.entryFy)
        self.get_content_area().add(self.entryFz)

    def on_button_clicked(self, widget):
        self.particleWin.updateField(
            self.entryFx.get_text(),
            self.entryFy.get_text(),
            self.entryFz.get_text()
        )


paramWindow = ParamWindow(particleWindow)
paramWindow.connect("destroy", Gtk.main_quit)
paramWindow.show_all()
Gtk.main()

glfwThread.join()
