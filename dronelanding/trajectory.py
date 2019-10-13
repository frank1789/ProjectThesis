#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# define constant and shortcut
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
        return self.M.reshape(1, 3)

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

    def __init__(self, shape, minradius, maxradius, step_size=1.0, increment=1.5):
        self.shape = []
        self.coordinate = []
        self.point_list = []
        self.highest_value = 0.0
        self.shape_type = shape
        self.build_shape(shape, minradius, maxradius, step_size, increment)

    def build_shape(self, shape, minradius, maxradius, step_size, increment):
        if shape == "hemisphere":
            print("Generate hemisphere minimum radius {:3.3f}, maximum radius {:3.3f}".format(
                minradius, maxradius))
            # generates a list with concentric rays
            radius_list = list(self.incremental_range(
                minradius, maxradius, step_size, increment))
            self.highest_value = max(radius_list)
            print("All radius values: {}, max radius: {}".format(
                radius_list, self.highest_value))
            for r in radius_list:
                self.shape.append(self.generate_hemisphere(r))
        elif shape == "cube":
            print("Generate trunk of a pyramid minimum radius {:3.3f}, maximum radius {:3.3f}".format(
                minradius, maxradius))
            # generates a list with concentric rays
            radius_list = list(self.incremental_range(
                minradius, maxradius, step_size, increment))
            self.highest_value = max(radius_list)
            print("All radius values: {}, max radius: {}".format(
                radius_list, self.highest_value))
            for r in radius_list:
                self.shape.append(self.generate_cube(r))
        else:
            raise ValueError(
                "Not valid input the shape must be: 'hemisphere', 'cube'.")

    @staticmethod
    def incremental_range(start, stop, step, inc):
        value = start
        while value < stop:
            yield value
            value += step
            step += inc
            print(value)

    @staticmethod
    def generate_hemisphere(radius):
        r = radius
        print("radius", radius)
        phi, theta = np.mgrid[0.0: 2 * PI: 20j, 0.0: -PI: 20j]
        _x = r * cos(phi) * cos(theta)
        _y = r * sin(phi) * cos(theta)
        _z = r * sin(theta)
        return _x, _y, _z

    @staticmethod
    def generate_cube(r):
        _x, _y = np.mgrid[-r/2: r/2: 20j, -r/2: r/2: 20j]
        _z = np.ones((20, 20)) * r
        return _x, _y, _z

    def translate(self):
        print("call translate")

    def rotate(self):
        print("call rotate")

    def get_coordinate(self):
        vect = []
        frame = 1
        self.__unpack_coordinate()
        if self.shape_type is "hemisphere":
            print("==> Centering the points cloud")
            trl_vect = np.array([0, 0, self.highest_value])
            for point in self.point_list:
                p = Transformation().translate(point, trl_vect).tolist()[0]
                vect.append(dict(name="IMG{:05d}".format(
                    frame), x=p[0], y=p[1], z=p[2]))
                frame += 1
        else:
            for point in self.point_list:
                p = point
                vect.append(dict(name="IMG{:05d}".format(
                    frame), x=p[0], y=p[1], z=p[2]))
                frame += 1

        return vect

    def __unpack_coordinate(self):
        for _x, _y, _z in self.shape:
            for x_i, y_i, z_i in zip(_x, _y, _z):
                for x_j, y_j, z_j in zip(x_i, y_i, z_i):
                    self.point_list.append([x_j, y_j, z_j])


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    camera_point = Trajectory("cube", 0.10, 30)
    camera_point2 = Trajectory("hemisphere", 0.10, 30)

    print(len(camera_point2.get_coordinate()))
    for j in camera_point.get_coordinate():
        print(j)
        name_file, x, y, z = j.values()
        ax.scatter(x, y, z)

    plt.tight_layout()
    plt.show()
