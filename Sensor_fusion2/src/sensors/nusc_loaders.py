# src/sensors/nusc_loaders.py

import os
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud


def get_ego_pose_xy_yaw(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    sd_token = next(iter(sample["data"].values()))
    sd = nusc.get("sample_data", sd_token)
    ego = nusc.get("ego_pose", sd["ego_pose_token"])

    x, y, _ = ego["translation"]
    yaw = Quaternion(ego["rotation"]).yaw_pitch_roll[0]
    return float(x), float(y), float(yaw)


def load_radar_xy(nusc, sample):
    token = sample["data"]["RADAR_FRONT"]
    return _load_sensor_xy(nusc, token)


def load_lidar_xy(nusc, sample):
    token = sample["data"]["LIDAR_TOP"]
    return _load_sensor_xy(nusc, token)


def load_camera_xy(nusc, sample):
    # camera는 box center만
    boxes = nusc.get_boxes(sample["data"]["CAM_FRONT"])
    return np.array([[b.center[0], b.center[1]] for b in boxes])


def _load_sensor_xy(nusc, sd_token):
    sd = nusc.get("sample_data", sd_token)
    path = os.path.join(nusc.dataroot, sd["filename"])

    if "RADAR" in sd["channel"]:
        pc = RadarPointCloud.from_file(path)
    else:
        pc = LidarPointCloud.from_file(path)

    pts = pc.points[:3, :].T
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    R = Quaternion(cs["rotation"]).rotation_matrix
    t = np.array(cs["translation"])

    ego = pts @ R.T + t
    return ego[:, :2]
