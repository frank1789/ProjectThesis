#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# define constant and shortcut
# from numpy.core._multiarray_umath import ndarray

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


class Transformation:
    def __init__(self):
        self.M = np.zeros((1, 3))

    def translate(self, vec, offset=np.zeros((3, 3))):
        """
        Tranlsate 'vect' by a certain offset
        :param vec: original point
        :param offset:  arrive point
        :return: translated vector
        """
        self.M += vec
        self.M += offset
        # return self.M.reshape(1, 3)
        return self

    def rotate(self, axis, vec, theta):
        """
        Rotate multidimensional array `X` `theta` degrees around axis `axis`
        :param axis: rotation axis
        :param vec:  original multidimensional array
        :param theta: deegree
        :return: rotated vector
        """
        c, s = np.cos(theta), np.sin(theta)
        if axis == 'X':
            self.M = np.dot(vec, np.array([
                [1., 0, 0],
                [0, c, -s],
                [0, s, c]
            ]))
        elif axis == 'Y':
            self.M = np.dot(vec, np.array([
                [c, 0, -s],
                [0, 1, 0],
                [s, 0, c]
            ]))
        elif axis == 'Z':
            self.M = np.dot(vec, np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1.],
            ]))

        return self.M


class Trajectory(object):
    shape = []
    coordinate = []

    def __init__(self, shape, minradius, maxradius, step_size=1.0, increment=1.5):
        self.build_shape(shape, minradius, maxradius, step_size, increment)

    @classmethod
    def build_shape(cls, shape, minradius, maxradius, step_size, increment):
        if shape == "hemisphere":
            print("Generate hemisphere minimum radius {:3.3f}, maximum radius {:3.3f}".format(
                minradius, maxradius))
            # generates a list with concentric rays
            radius_list = list(cls.incremental_range(
                minradius, maxradius, step_size, increment))
            cls.lowest_value = max(radius_list)
            print("All radius values: {}, max radius: {}".format(
                radius_list, cls.lowest_value))
            for r in radius_list:
                cls.shape.append(cls.generate_hemisphere(r))
        elif shape == "pyramid":
            print("Generate trunk of a pyramid minimum radius {:3.3f}, maximum radius {:3.3f}".format(
                minradius, maxradius))
            # generates a list with concentric rays
            radius_list = list(cls.incremental_range(
                minradius, maxradius, step_size, increment))
            cls.lowest_value = max(radius_list)
            print("All radius values: {}, max radius: {}".format(
                radius_list, cls.lowest_value))
            for r in radius_list:
                cls.shape.append(cls.generate_trunk_pyramid(-r, r, r))
        else:
            raise ValueError(
                "Not valid input the shape must be: Hemisphere, ecc.")

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
        y = r * sin(phi) * cos(theta)
        z = r * sin(theta)
        return x, y, z

    @classmethod
    def generate_trunk_pyramid(cls, min_size, max_size, r):
        x, y, z = np.mgrid[min_size: max_size: 20j,
                           min_size * 1.20: max_size * 1.20: 20j,
                           0: r: 20j]
        return x, y, z

    #     X, Y, Z = np.meshgrid(np.arange(0, size, 1),
    #                   np.arange(0, size, 1),
    #                   np.arange(0, size, 1))

    def get_translation(self):
        for x, y, z in self.shape:
            for x_i, y_i, z_i in zip(x, y, z):
                for x_j, y_j, z_j in zip(x_i, y_i, z_i):
                    vector_list = [x_j, y_j, z_j]
                    #vr = Transformation().translate(
                    #    vector_list, np.array([0, 0, self.lowest_value])).rotate('Y', vector_list, PI/6).tolist()
                    self.coordinate.append(dict(x=vector_list[0], y=vector_list[1], z=vector_list[2]))
        return self.coordinate


if __name__ == "__main__":
    # Set colours and render
    Transformation()
    # a()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    camera_point = Trajectory("pyramid", 0.10, 30)
    print(len(camera_point.get_translation()))
    for j in camera_point.get_translation():
       # print(j)
        x, y, z = j.values()
        ax.scatter(x, y, z)

    plt.tight_layout()
    plt.show()

    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])
    # ax.set_zlim([-1,1])
    ax.set_aspect('equal')
