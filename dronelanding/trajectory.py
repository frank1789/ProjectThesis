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

    def translate(self, vec, offset=np.zeros((1, 3))):
        """
        Translate 'vect' by a certain offset
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
            raise ValueError(
                "Error: density value cannot be negative or float number.")
        self.density = density
        if lower_limit < (0.0 or 0):
            raise ValueError("Error: lower limit cannot be a negative number.")
        if upper_limit < (0.0 or 0):
            raise ValueError("Error: upper limit cannot be a negative number.")
        if step_size < (0.0 or 0):
            raise ValueError("Error: step size cannot be a negative number.")
        if increment < (0.0 or 0):
            raise ValueError("Error: increment cannot be a negative number.")
        self.build_shape(shape, lower_limit, upper_limit, step_size, increment)
        self.__unpack_coordinate()

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
                self.shape.append(self.generate_cube(lower, edge, self.density))
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
    def generate_cube(edge, heigth, n):
        """
        Generate a truncated pyramid.
        :param heigth:
        :param edge: (float) edge length.
        :param n: (int) number of points that must be generated to fill the previously provided dimensions.
        :return: (np.array) coordinate of each points.
        """
        _x, _y = np.mgrid[-edge/2: edge/2: n * 1j, -edge/2: edge/2: n * 1j]
        if heigth > 1.5:
            factor = heigth/2 * 1.20
            _x += np.random.uniform(-factor, factor, size=(n, n))
            _y += np.random.uniform(-factor, factor, size=(n, n))
        else:
            _x += np.random.uniform(-0.5, 0.5, size=(n, n))
            _y += np.random.uniform(-0.5, 0.5, size=(n, n))
        _z = np.ones((20, 20)) * heigth
        return _x, _y, _z

    def get_coordinate(self, **kwargs):
        vect = []
        frame = 1
        p = None
        for p in self.coordinate:
            # center hemisphere in (0, 0)
            if self.shape_type is "hemisphere":
                p = (lambda x: Transformation().translate(
                    x, np.array([0, 0, self.highest_value])).tolist()[0])(p)

            # apply transformation if in arguments
            if kwargs.__len__() is not 0:
                for key, value in kwargs.items():
                    if key is "translate":
                        p = (lambda x: Transformation().translate(
                            x, np.array(value[0:])).tolist()[0])(p)
                    elif key is "rotate":
                        axis, angle = value.items()
                        p = (lambda x, axis, angle: Transformation().rotate(
                            axis, p, angle).tolist())(np.array(p), axis[1], angle[1])
                    else:
                        raise ValueError(
                            "Error: the argument must be 'translate=[x,y,z]' or 'rotate={'axis':'Z', 'angle':PI/6}'")

            # append value output vector
            if len(p) < 3:
                p = [0, 0, p]
            vect.append(dict(filename="IMG", x=p[0], y=p[1], z=p[2]))
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
                    self.coordinate.append([x_j, y_j, z_j])


if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    camera_point = Trajectory("cube", 0.375, 30.0, 5)
    # camera_point2 = Trajectory("hemisphere", 0.10, 30, 10)

    # print(len(camera_point2.get_coordinate()))
    print(len(camera_point.get_coordinate()))
    a = dict(axis="X", angle=PI)
    b = dict(axis="Z", angle=PI / 6)
    for j in camera_point.get_coordinate():
        name_file, x, y, z = j.values()
        ax.scatter(x, y, z)

    plt.tight_layout()
    plt.show()
