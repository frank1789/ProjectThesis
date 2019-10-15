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

    def __init__(self, shape, lower_limit, upper_limit, density, step_size=1.0, increment=1.5):
        """
        The trajectory class constructs a point cloud on a geometric shape. At the moment it accepts that of an
        hemisphere and a cube (it returns a truncated pyramid).
        :param shape: (str) the shape to be obtained, accepts hemisphere and cube.
        :param lower_limit: (float) lower dimension which must take the geometric form (radius, side, etc.).
        :param upper_limit: (float)upper dimension which must take the geometric form (radius, side, etc.).
        :param density: (int) number of points that must be generated to fill the previously provided dimensions.
        :param step_size: (float) indicates the step between the minimum and maximum size, optional.
        :param increment: (float) indicates the increment between the next and previous step between the minimum and
        maximum size, optional.
        """
        self.shape = []
        self.coordinate = []
        self.highest_value = 0.0
        self.shape_type = shape
        if density < 1 or type(density) is not int:
            raise ValueError("Error: density value cannot be negative or float number.")
        self.density = density
        if lower_limit < 0.0:
            raise ValueError("Error: lower limit cannot be a negative number.")
        if upper_limit < 0.0:
            raise ValueError("Error: upper limit cannot be a negative number.")
        if step_size < 0.0:
            raise ValueError("Error: step size cannot be a negative number.")
        if increment < 0.0:
            raise ValueError("Error: increment cannot be a negative number.")
        self.build_shape(shape, lower_limit, upper_limit, step_size, increment)

    def build_shape(self, shape, lower, upper, step_size, increment):
        """
        the build function checks that the correct shape has been requested and builds it otherwise raises an error due
        to incorrect value entered.
        :param shape: (str) the shape to be obtained, accepts hemisphere and cube
        :param lower: (float) lower dimension which must take the geometric form (radius, side, etc.).
        :param upper: (float)upper dimension which must take the geometric form (radius, side, etc.).
        :param step_size: (float) indicates the step between the minimum and maximum size, optional.
        :param increment: (float) indicates the increment between the next and previous step between the minimum and
        maximum size, optional.
        """
        if shape == "hemisphere":
            print("Generate hemisphere minimum radius {:3.3f}, maximum radius {:3.3f}".format(
                lower, upper))
            # generates a list with concentric rays
            radius_list = list(self.incremental_range(
                lower, upper, step_size, increment))
            self.highest_value = max(radius_list)
            print("All radius values: {}, max radius: {}".format(
                radius_list, self.highest_value))
            for r in radius_list:
                self.shape.append(self.generate_hemisphere(r, self.density))
        elif shape == "cube":
            print("Generate cube minimum length edge {:3.3f}, maximum length edge {:3.3f}".format(
                lower, upper))
            # generates a list with concentric rays
            edges = list(self.incremental_range(
                lower, upper, step_size, increment))
            self.highest_value = max(edges)
            print("All radius values: {}, max radius: {}".format(
                edges, self.highest_value))
            for edge in edges:
                self.shape.append(self.generate_cube(edge, self.density))
        else:
            raise ValueError(
                "Not valid input the shape must be: 'hemisphere', 'cube'.")

    @staticmethod
    def incremental_range(start, stop, step, inc):
        """
        the function calculates a sequence with constant or variable pitch.
        :param start: (float) value to start.
        :param stop:  (float) value to reach.
        :param step:  (float) indicates the step between the minimum and maximum size.
        :param inc: (float) indicates the increment between the next and previous step between the minimum and
        maximum size.
        :return: (float) value's sequence.
        """
        value = start
        while value < stop:
            yield value
            value += step
            step += inc
            print(value)

    @staticmethod
    def generate_hemisphere(radius, n):
        """
        Generate a hemisphere.
        :param radius: (float) radius length.
        :param n: (int) number of points that must be generated to fill the previously provided dimensions.
        :return: (np.array) coordinate of each points.
        """
        print("radius", radius)
        phi, theta = np.mgrid[0.0: 2 * PI: n * 1j, 0.0: -PI: n * 1j]
        _x = radius * cos(phi) * cos(theta)
        _y = radius * sin(phi) * cos(theta)
        _z = radius * sin(theta)
        return _x, _y, _z

    @staticmethod
    def generate_cube(edge, n):
        """
        Generate a truncated pyramid.
        :param edge: (float) edge length.
        :param n: (int) number of points that must be generated to fill the previously provided dimensions.
        :return: (np.array) coordinate of each points.
        """
        _x, _y = np.mgrid[-edge / 2: edge / 2: n * 1j, -edge / 2: edge / 2: n * 1j]
        _z = np.ones((20, 20)) * edge
        return _x, _y, _z

    def translate(self):
        print("call translate")

    def rotate(self):
        print("call rotate")

    def get_coordinate(self):
        vect = []
        point_list = []
        frame = 1
        point_list.append(self.__unpack_coordinate())
        if self.shape_type is "hemisphere":
            print("==> Centering the points cloud")
            trl_vect = np.array([0, 0, self.highest_value])
            for point in point_list:
                p = Transformation().translate(point, trl_vect).tolist()[0]
                vect.append(dict(name="IMG{:05d}".format(
                    frame), x=p[0], y=p[1], z=p[2]))
                frame += 1
        else:
            for point in point_list:
                p = point
                vect.append(dict(name="IMG{:05d}".format(
                    frame), x=p[0], y=p[1], z=p[2]))
                frame += 1

        return vect

    def __unpack_coordinate(self):
        """
        Transform np.array in list
        :return: (list) coordinate [x, y, z]
        """
        for _x, _y, _z in self.shape:
            for x_i, y_i, z_i in zip(_x, _y, _z):
                for x_j, y_j, z_j in zip(x_i, y_i, z_i):
                    return [x_j, y_j, z_j]


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    camera_point = Trajectory("cube", 0.10, 30, 10)
    camera_point2 = Trajectory("hemisphere", 0.10, 30, 20)

    print(len(camera_point2.get_coordinate()))
    print(len(camera_point.get_coordinate()))
    for j in camera_point2.get_coordinate():
        print(j)
        name_file, x, y, z = j.values()
        ax.scatter(x, y, z)

    plt.tight_layout()
    plt.show()

    Trajectory("cube", 0.1, 30, 1)
