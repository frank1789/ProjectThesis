import os
import sys
from operator import itemgetter

import bpy
import numpy as np

from traje import RaspberryPiCamera, Transformation


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
        bpy.data.scenes['Scene'].render.filepath = '/Users/francesco/PycharmProjects/ProjectThesis/dronelanding/result/IMG_{:d}.jpg'.format(
            frame_count)
        bpy.ops.render.render(write_still=True)
