#!usr/bin/env python3
# -*- conding: utf-8 -*-

import unittest

from raspberrycamera import RaspberryPiCamera
from trajectory import PI, rad2degree, degree2rad, Trajectory


# own test class


class TestSupportFunction(unittest.TestCase):
    def test_rad2degree(self):
        self.assertEqual(rad2degree(0.17453292519943295), 10,
                         "given 0.1745 rad should be return 10 degrees")

    def test_degree2rad(self):
        self.assertEqual(degree2rad(90), PI / 2,
                         "given 90 degrees should be return PI/2 rad")

    def test_camera_constant_resolution_x(self):
        self.assertEqual(RaspberryPiCamera().getResolution_x(), 3280)

    def test_camera_constant_resolution_y(self):
        self.assertEqual(RaspberryPiCamera().getResolution_y(), 2464)

    def test_camera_constant_focal_length(self):
        self.assertEqual(RaspberryPiCamera().getFocal_length(), 3.04)

    def test_camera_constant_sensor_size(self):
        self.assertEqual(RaspberryPiCamera().getSensor_size(), 4.60)

    def test_camera_constant_f_number(self):
        self.assertEqual(RaspberryPiCamera().getFnumber(), 2.0)

    def test_trajectory_lower_value(self):
        self.assertRaises(ValueError, Trajectory, "cube", -15.0, 100.5, -3)

    def test_trajectory_upper_value(self):
        self.assertRaises(ValueError, Trajectory, "cube", 0.0, -30, -3)

    def test_trajectory_desnity_value(self):
        self.assertRaises(ValueError, Trajectory, "cube", 0.0, 30, -7)

    def test_trajectory_density_type(self):
        self.assertRaises(ValueError, Trajectory, "cube", 0.0, 15, 3.5)

    def test_trajectory_step_value(self):
        self.assertRaises(ValueError, Trajectory, "cube", 0.1, 25, 5, -0.25)

    def test_trajectory_increment_value(self):
        self.assertRaises(ValueError, Trajectory, "cube", 0.1, 35, 7, 0.25, -1)

    def test_trajectory_shape_string(self):
        self.assertRaises(ValueError, Trajectory, "cone", 0.0, 30, 3)

    def test_trajectory_size_cube(self):
        camera_point_1 = Trajectory("cube", 0.10, 30, 5)
        camera_point_2 = Trajectory("cube", 0.10, 30, 10)
        camera_point_3 = Trajectory("cube", 0.10, 30, 30)

        len_1 = len(camera_point_1.get_coordinate())
        len_2 = len(camera_point_2.get_coordinate())
        len_3 = len(camera_point_3.get_coordinate())
        self.assertEqual(camera_point_1.get_coordinate(), len_1)
        self.assertEqual(camera_point_2.get_coordinate(), len_2)
        self.assertEqual(camera_point_3.get_coordinate(), len_3)

    def test_trajectory_size_hempishere(self):
        camera_point_1 = Trajectory("hemisphere", 0.10, 30, 5)
        camera_point_2 = Trajectory("hemisphere", 0.10, 30, 10)
        camera_point_3 = Trajectory("hemisphere", 0.10, 30, 30)

        len_1 = len(camera_point_1.get_coordinate())
        len_2 = len(camera_point_2.get_coordinate())
        len_3 = len(camera_point_3.get_coordinate())
        self.assertEqual(camera_point_1.get_coordinate(), len_1)
        self.assertEqual(camera_point_2.get_coordinate(), len_2)
        self.assertEqual(camera_point_3.get_coordinate(), len_3)


if __name__ == "__main__":
    unittest.main()
