#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
