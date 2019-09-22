import numpy

from OpenGL.GL import glViewport

from pyrr import matrix44, quaternion

class Projection:
    def __init__(self, distance):
        self.distance = distance
        self.ratio    = 4./3.
        self.update()

    def update(self):
        projection = matrix44.create_perspective_projection(20.0, self.ratio, 0.1, 1000.0)
        look = matrix44.create_look_at(
            eye    = [0, -self.distance, 0],
            target = [0, 0, 0],
            up     = [0, 0, -1])

        self.matrix = numpy.matmul(look, projection)

    def update_ratio(self, width, height, update_viewport = True):
        if update_viewport:
            glViewport(0,0,width,height)

        self.ratio = width/height
        self.update()

    def update_distance(self, change):
        self.distance += change
        self.update()

    def get(self):
        return self.matrix

class Rotation:
    def __init__(self, shift, x = numpy.pi, z = numpy.pi):
        self.matrix = matrix44.create_from_translation(shift),
        self.rotation_x = quaternion.Quaternion()
        self.update(x,z)

    def update(self, x, z):
        rotation_x = quaternion.Quaternion(quaternion.create_from_eulers([x,0,0]))
        rotation_z = self.rotation_x.conjugate.cross(
                quaternion.Quaternion(quaternion.create_from_eulers([0,0,z])))
        self.rotation_x = self.rotation_x.cross(rotation_x)

        self.matrix = numpy.matmul(
            self.matrix,
            matrix44.create_from_quaternion(rotation_z.cross(self.rotation_x))
        )
        self.inverse_matrix = numpy.linalg.inv(self.matrix)

    def get(self):
        return self.matrix

    def get_inverse(self):
        return self.inverse_matrix


