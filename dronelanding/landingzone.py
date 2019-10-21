#!usr/bin/env python3
# -*- coding: utf-8 -*-

from trajectory import Trajectory


class DayTimeCycle(object):
    def __init__(self) -> None:
        self.sunrise = {'period': "sunrise", 'color': (218, 97, 41), 'temp_colour': 1.5, 'azimut': 78.0, 'zenit': 0.0}
        self.sunset = {'period': "sunset", 'color': (226, 128, 48), 'temp_colour': 1.8, 'azimut': -62.6, 'zenit': 0.0}
        self.mid_morning = {'period': "mid_morning", 'color': (254, 230, 208), 'temp_colour': 6.6, 'azimut': 43.2,
                            'zenit': 0.0}
        self.clear_sky = {'period': "clear_sky", 'color': (255, 255, 255), 'temp_colour': 9.8, 'azimut': 0.0,
                          'zenit': 0.0}
        self.mid_afternoon = {'period': "mid_afternoon", 'color': (255, 155, 130), 'temp_colour': 6.8, 'azimut': -34.8,
                              'zenit': 0.0}

    def daytime(self) -> list:
        return [self.sunrise, self.mid_morning, self.clear_sky, self.mid_afternoon, self.sunset]


class PositionMate(object):
    def __init__(self, model_mate) -> None:
        if model_mate is "RedLanding":
            self.position_landing_zone = [
                {'position': "road_cross", 'coordinate': (0.74, 6.5, 0.12)},
                {'position': "grass_sidewalk", 'coordinate': (5.47, 19.04, 0.12)},
                {'position': "grass", 'coordinate': (28.782, 59.297, 0.12)},
                {'position': "road", 'coordinate': (1.88, 41.3, 0.12)},
            ]
        elif model_mate is "CiterX":
            self.position_landing_zone = [
                {'position': "road_cross", 'coordinate': (0.74, 6.5, 0.12)},
                {'position': "grass_sidewalk", 'coordinate': (5.47, 19.04, 0.12)},
                {'position': "grass", 'coordinate': (28.782, 59.297, 0.12)},
                {'position': "road", 'coordinate': (1.88, 41.3, 0.12)},
            ]

        elif model_mate is "GreenSquare":
            self.position_landing_zone = [
                {'position': "road_cross", 'coordinate': (0.74, 6.5, 0.02)},
                {'position': "grass_sidewalk", 'coordinate': (5.47, 19.04, 0.02)},
                {'position': "grass", 'coordinate': (28.782, 59.297, 0.02)},
                {'position': "road", 'coordinate': (1.88, 41.3, 0.02)},
            ]

    def position(self) -> list:
        return self.position_landing_zone


class SetupSceneObject(object):
    def __init__(self, model_mate, shape, lower_limit, upper_limit, density, step_size=1.0, increment=1.5):
        self._camera_trajectory = Trajectory(shape, lower_limit, upper_limit, density, step_size, increment)
        self._position = PositionMate(model_mate)
        self._daytimecycle = DayTimeCycle().daytime()
        self.scenes = self.__setup_scenes()

    def __setup_scenes(self) -> list:
        counter = 1
        _setup = None
        collection = []
        for pose in self._position.position():
            for daytime in self._daytimecycle:
                _points = self._camera_trajectory.get_coordinate(translate=pose['coordinate'])
                for point in _points:
                    # merge temporary dict merge position mate and daytime
                    _temp = self.__merge(pose, daytime)
                    # merge point camera dict with temporary and update name file
                    _setup = self.__merge(point, _temp)
                    _setup['filename'] += "{:05d}".format(counter)
                    collection.append(_setup)
                    counter += 1
        # return all parametrized scenes
        return collection

    @staticmethod
    def __merge(dict1, dict2) -> dict:
        res = {**dict1, **dict2}
        return res

    @property
    def get_setup(self) -> list:
        return self.scenes

