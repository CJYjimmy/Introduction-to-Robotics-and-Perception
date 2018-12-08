

import numpy as np
import math
import json



data = json.load(open('map_arena.json'))
WEIGHT = data['width']
HEIGHT = data['height']
INCHE_IN_MILLIMETERS = data["scale"]



def recognize_cube(cubePose):
    x, y = cubePose
    if x < WEIGHT/2:
        if y < HEIGHT/2:
            return 'C'
        else:
            return 'A'
    else:
        if y < HEIGHT/2:
            return 'D'
        else:
            return 'B'


def normalize_degree(degree):
    while degree > 180:
        degree -= 360
    while degree < -180:
        degree += 360
    return degree


def rotate_point(x, y, heading_deg):
    c = math.cos(math.radians(heading_deg))
    s = math.sin(math.radians(heading_deg))
    xr = x * c + y * -s
    yr = x * s + y * c
    return xr, yr

class CoordinateTransformer():

    def __init__(self, originPose, mapPose):
        self.T = np.zeros((3,3))
        self.dh = normalize_degree(mapPose[2] - originPose[2])
        self.T[0,0] = math.cos(self.dh / 180 * PI)
        self.T[0,1] = - math.sin(self.dh / 180 * PI)
        self.T[1,0] = math.sin(self.dh / 180 * PI)
        self.T[1,1] = math.cos(self.dh / 180 * PI)
        self.T[2,2] = 1

        mapPose_homo = np.array([[mapPose[0]], [mapPose[1]], [1]])
        originPose_homo = np.array([[originPose[0]], [originPose[1]], [1]])
        t = mapPose_homo * INCHE_IN_MILLIMETERS - self.T.dot(originPose_homo)

        self.T[:,2] = t.reshape((-1,))
        self.T[2,2] = 1

    def map_to_origin(self, mapPose):
        homo_p_map = np.ones((3,1))
        homo_p_map[0,0] = mapPose[0] * INCHE_IN_MILLIMETERS
        homo_p_map[1,0] = mapPose[1] * INCHE_IN_MILLIMETERS
        homo_p_origin = np.linalg.inv(self.T).dot(homo_p_map)
        return (
            homo_p_origin[0,0],
            homo_p_origin[1,0],
            normalize_degree(mapPose[2] - self.dh)
        )

    def origin_to_map(self, originPose):
        homo_p_origin = np.ones((3,1))
        homo_p_origin[0,0] = originPose[0]
        homo_p_origin[1,0] = originPose[1]
        homo_p_map = self.T.dot(homo_p_origin)
        mapPose = (
            homo_p_map[0,0] / INCHE_IN_MILLIMETERS,
            homo_p_map[1,0] / INCHE_IN_MILLIMETERS,
            normalize_degree(originPose[2] + self.dh)
        )
        return mapPose