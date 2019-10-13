#!usr/bin/env python3
# -*- conding: utf-8 -*-

import unittest
from trajectory import PI, rad2degree, degree2rad
from raspberrycamera import RaspberryPiCamera

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


if __name__ == "__main__":
    unittest.main()
