from OpenGL.GLUT import *

class MouseDragMonitor:
    def __init__(self, button, callback):
        self.button   = button
        self.callback = callback
        self.active   = False

    def on_mouse(self, button, state, x, y):
        if button == self.button:
            self.active = (state == GLUT_DOWN)
            self.last_x = x
            self.last_y = y

    def on_mouse_move(self, x, y):
        if self.active:
            delta_x = self.last_x - x
            delta_y = y - self.last_y
            self.last_x = x
            self.last_y = y
            self.callback(delta_x, delta_y)
