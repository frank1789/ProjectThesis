#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# define constant and shortcut
from numpy.core._multiarray_umath import ndarray

PI = np.pi
cos = np.cos
sin = np.sin


def degree2rad(angle):
    """
    Given an angle expressed in degrees it returns an angle expressed in radians.
    :param angle:
    """
    return angle * PI / 180


def rad2degree(angle):
    """
    Given an angle expressed in radians it returns an angle expressed in degrees.
    :param angle:
    """
    return angle * 180 / PI


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


# generate sequence of angle
# alpha = range(0, 360, 10)


# # my test
# class TestSupportFunction(unittest.TestCase):

#     def test_rad2degree(self):
#         self.assertEqual(rad2degree(0.17453292519943295), 10, "given 0.1745 rad should be return 10 degrees")

#     def test_degree2rad(self):
#         self.assertEqual(degree2rad(90), PI / 2, "given 90 degrees should be return PI/2 rad")


# ########################################################################################################################

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
        lowest_point = 0.0
        for x, y, z in self.hemisphere:
            for x_i, y_i, z_i in zip(x, y, z):
                for x_j, y_j, z_j in zip(x_i, y_i, z_i):
                    vector_list = [x_j, y_j, z_j]
                    if not (-0.1 > x_j > 0.1) or not (-0.1 > y_j > 0.1):
                        # translate hemisphere
                        # pass
                    #else:
                        vr = TransformationMatrix().translate(vector_list, np.array([0, 0, self.lowest_value])).tolist()[0]
                        self.coordinate.append(dict(x=vr[0], y=vr[1], z=vr[2]))
        return self.coordinate


if __name__ == "__main__":

    # unittest.main()

    # Set colours and render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    camera_point = Trajectory(0.10, 30)
    print(len(camera_point.get_translation()))
    for j in camera_point.get_translation():
        print(j)
        x, y, z = j.values()

        # x, y, z =

        ax.scatter(x, y, z)

    plt.tight_layout()
    plt.show()

    # ax.scatter(xx,yy,zz,color="k",s=20)

    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])
    # ax.set_zlim([-1,1])
    # ax.set_aspect('equal')

    # def incremental_range(start, stop, step, inc):
    #     value = start
    #     while value < stop:
    #         yield value
    #         value += step
    #         step += inc
    #         print(value)
    #
    # print(list(incremental_range(0, 30, 0.05, 0.20)))
    # print(len(list(incremental_range(0, 30, 0.05, 0.20))))
