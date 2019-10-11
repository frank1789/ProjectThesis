import os
import sys
from operator import itemgetter

import bpy
import numpy as np



class RaspberryPiCamera:
    def __init__(self):
        self.RESOLUTION_X = 3280
        self.RESOLUTION_Y = 2464
        self.FOCAL_LENGTH = 3.04
        self.SENSOR_SIZE = 4.60
        self.F = 2.0

    def getResolution_x(self):
        return self.RESOLUTION_X

    def getResolution_y(self):
        return self.RESOLUTION_Y

    def getFocal_length(self):
        return self.FOCAL_LENGTH

    def getSensor_size(self):
        return self.SENSOR_SIZE

    def getFnumber(self):
        return self.F


# setup camera properties
def setup_camera():
    camera = None
    # check for 1st camera
    if (len(bpy.data.cameras) == 1):
        camera = bpy.data.objects['Camera']

    # modify properties
    camera.data.lens_unit = 'MILLIMETERS'
    camera.data.lens = RaspberryPiCamera().getFocal_length()
    camera.data.dof.aperture_fstop = RaspberryPiCamera().getFnumber()
    camera.data.sensor_width = RaspberryPiCamera().getSensor_size()

    return camera


def setup_output():
    # setup resolution scene output
    bpy.context.scene.render.resolution_x = RaspberryPiCamera().getResolution_x()
    bpy.context.scene.render.resolution_y = RaspberryPiCamera().getResolution_y()

    # setup image setting
    bpy.context.scene.render.image_settings.file_format = 'JPEG'


#
#
#
#
#
#
# define constant and shortcut
PI = np.pi
cos = np.cos
sin = np.sin

class TransformationMatrix:

    def __init__(self):
        self.M = np.zeros((1, 3))

    def translate(self, vec, offset=np.zeros((3,3))):
        """
        Tranlsate 'vect' by a certain offset
        :param vec: original point
        :param offset:  arrive point
        :return: translated vector
        """
        self.M += vec
        self.M += offset
        return self.M.reshape(1,3)

    @classmethod
    def rotate(cls, axis, vec, theta):
        """
        Rotate multidimensional array `X` `theta` degrees around axis `axis`
        :param axis: rotation axis
        :param vec:  original multidimensional array
        :param theta: deegree
        :return: rotated vector
        """
        c, s = np.cos(theta), np.sin(theta)
        if axis == 'X':
            cls.M = np.dot(vec, np.array([
                [1., 0, 0],
                [0, c, -s],
                [0, s, c]
            ]))
        elif axis == 'Y':
            cls.M = np.dot(vec, np.array([
                [c, 0, -s],
                [0, 1, 0],
                [s, 0, c]
            ]))
        elif axis == 'Z':
            cls.M = np.dot(vec, np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1.],
            ]))

        return cls.M









#
#


class Trajectory(object):
    def __init__(self, minradius, maxradius, step_size=1.5, increment=2.5):
        print("Generate hemisphere minimum radius {:3.3f}, maximum radius {:3.3f}".format(minradius, maxradius))
        self.min_radius = minradius
        self.max_radius = maxradius
        self.coordinate = []
        self.hemisphere = []
        self.radius = list(self.incremental_range(self.min_radius, self.max_radius, step_size, increment))
        self.lowest_value = max(self.radius)
        for i in self.radius:
            self.hemisphere.append(self.generate_hemisphere(i))

    @staticmethod
    def incremental_range(start, stop, step, inc):
        value = start
        while value < stop:
            yield value
            value += step
            step += inc
            print(value)


    @classmethod
    def generate_hemisphere(cls, radius):
        r = radius
        print("radius", radius)
        phi, theta = np.mgrid[0.0: 2 * PI: 20j, 0.0: -PI: 20j]
        x = r * cos(phi) * cos(theta)
        y= r * sin(phi) * cos(theta)
        z = r *sin(theta)
        return x, y, z

    def get_translation(self):
        for x, y, z in self.hemisphere:
            for x_i, y_i, z_i in zip(x, y, z):
                for x_j, y_j, z_j in zip(x_i, y_i, z_i):
                    vector_list = [x_j, y_j, z_j]
                    if (-0.1 > x_j > 0.1) and (-0.1 > y_j > 0.1):
                        # translate hemisphere
                        pass
                    else:
                        vr = \
                        TransformationMatrix().translate(vector_list, np.array([0, 0, self.lowest_value])).tolist()[0]
                        self.coordinate.append(dict(x=vr[0], y=vr[1], z=vr[2]))
        return self.coordinate



if __name__ == '__main__':
    # prepare camera setting as Raspberry Pi camera V2
    camera = setup_camera()
    # initialize key-frame counter
    frame_count = 1
    # generate points cloud
    traject_points = Trajectory(0.01, 30)
    points_cloud = traject_points.get_translation()

    # set end frame
    bpy.context.scene.frame_end = len(points_cloud)

    # set position and keyframe
    for point in points_cloud:
        x, y, z = point.values()
        # set camera translation
        camera.location.x = x
        camera.location.y = y
        camera.location.z = z
        # set camera rotation
        camera.rotation_euler[0] = 0
        camera.rotation_euler[1] = 0
        camera.rotation_euler[2] = 0
        # update frame counter
        frame_count += 1
        bpy.data.scenes['Scene'].render.filepath = '/Users/francesco/PycharmProjects/ProjectThesis/dronelanding/result/IMG_{:d}.jpg'.format(frame_count)
        bpy.ops.render.render( write_still=True )
