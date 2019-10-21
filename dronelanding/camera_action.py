import os
import sys

import bpy

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

from raspberrycamera import RaspberryPiCamera
from trajectory import degree2rad
from landingzone import SetupSceneObject


# setup camera properties
def setup_camera():
    camera = None
    # check for 1st camera
    if len(bpy.data.cameras) == 1:
        camera = bpy.data.objects['Camera']

    # modify properties
    camera.data.lens_unit = 'MILLIMETERS'
    camera.data.lens = RaspberryPiCamera().getFocal_length()
    camera.data.dof.aperture_fstop = RaspberryPiCamera().getFnumber()
    camera.data.sensor_width = RaspberryPiCamera().getSensor_size()
    return camera


def setup_output():
    # setup resolution scene output
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    # setup image setting
    bpy.context.scene.render.image_settings.file_format = 'JPEG'


if __name__ == '__main__':
    # prepare camera setting as Raspberry Pi camera V2
    camera = setup_camera()
    # setup sun light
    sun = bpy.data.objects['Light']
    # detect configuration
    mate_type = None
    obj_mate = None
    workspace_file = bpy.path.basename(bpy.context.blend_data.filepath)
    if workspace_file == 'citerx_landing_zone.blend':
        mate_type = "CiterX"
        obj_mate = bpy.data.objects['CircleLandingZone']
    elif workspace_file == 'orange_landing_zone.blend':
        mate_type = "RedLanding"
        obj_mate = bpy.data.objects['CircleLandingZone']
    elif workspace_file == 'square_landing_zone.blend':
        mate_type = "GreenSquare"
        obj_mate = bpy.data.objects['SquareLandingZone']
    # confiure save path 
    savepath = os.path.join('//landingzone/{:s}'.format(mate_type), mate_type)
    bpy.data.scenes['Scene'].render.filepath = savepath
    # build configuration
    configuration = SetupSceneObject(mate_type, "cube", 0.375, 30.0, 5)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(configuration.get_setup)
    for ob in scene.objects:
        for counterframe, config in enumerate(configuration.get_setup):
            # setup cycle daytime
            sun.data.energy = config['temp_colour']
            sun.data.color = tuple(map(lambda x: x / 255, config['color']))
            # positioning sun at 100m height
            sun.location[2] = 100.0
            # set orientation sun
            sun.rotation_euler[0] = 0.0  # x axis
            sun.rotation_euler[1] = degree2rad(config['azimut'])  # y axis
            sun.rotation_euler[2] = config['zenit']  # z axis
            # setup mate landing zone
            x, y, z = config['coordinate']
            obj_mate.location.x = x
            obj_mate.location.y = y
            obj_mate.location.z = z
            # setup camera roto-translation
            camera.location.x = config['x']
            camera.location.y = config['y']
            camera.location.z = config['z']
            # setup camera rotation
            camera.rotation_euler[0] = 0.0
            camera.rotation_euler[1] = 0.0
            camera.rotation_euler[2] = 0.0
            # render export and save file
            #bpy.ops.render.render(write_still=True)
            ob.keyframe_insert(data_path='location', frame=counterframe)
