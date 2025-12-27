# src/sensors/nusc_loaders.py
from __future__ import annotations

import numpy as np
from typing import Tuple

from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix


def _pc_to_ego(nusc, sample_data_token: str, pc_cls) -> np.ndarray:
    """
    Load pointcloud (lidar/radar) and transform to ego vehicle frame.
    x forward, y left, z up (nuScenes ego frame)
    """
    sd = nusc.get("sample_data", sample_data_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    # file path
    fpath = nusc.get_sample_data_path(sample_data_token)

    # load points in sensor frame
    pc = pc_cls.from_file(fpath)

    # sensor -> ego  (rotation MUST be Quaternion)
    T_s2e = transform_matrix(
        cs["translation"],
        Quaternion(cs["rotation"]),
        inverse=False
    )
    pc.transform(T_s2e)

    return pc.points  # (D, N)


def load_lidar_xy(nusc, sample_token: str, sensor_channel: str = "LIDAR_TOP") -> np.ndarray:
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"].get(sensor_channel, None)
    if sd_token is None:
        return np.zeros((0, 2), dtype=np.float32)

    pts = _pc_to_ego(nusc, sd_token, LidarPointCloud)
    xy = pts[:2, :].T.astype(np.float32)
    return xy


def load_radar_xy_vxy(
    nusc,
    sample_token: str,
    sensor_channel: str = "RADAR_FRONT",
    max_range_m: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray]:
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"].get(sensor_channel, None)
    if sd_token is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    pts = _pc_to_ego(nusc, sd_token, RadarPointCloud)

    x = pts[0, :]
    y = pts[1, :]

    # raw velocity
    vx = pts[6, :] if pts.shape[0] > 7 else np.zeros_like(x)
    vy = pts[7, :] if pts.shape[0] > 7 else np.zeros_like(x)

    # compensated velocity if available
    if pts.shape[0] > 10:
        vx_c = pts[8, :]
        vy_c = pts[9, :]
        use_comp = ~np.isnan(vx_c) & ~np.isnan(vy_c)
        vx = np.where(use_comp, vx_c, vx)
        vy = np.where(use_comp, vy_c, vy)

    xy = np.stack([x, y], axis=1).astype(np.float32)
    vxy = np.stack([vx, vy], axis=1).astype(np.float32)

    r = np.linalg.norm(xy, axis=1)
    keep = r < float(max_range_m)
    return xy[keep], vxy[keep]


def load_lidar_xyz(nusc, sample_token: str, sensor_channel: str = "LIDAR_TOP") -> np.ndarray:
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"].get(sensor_channel, None)
    if sd_token is None:
        return np.zeros((0, 3), dtype=np.float32)

    pts = _pc_to_ego(nusc, sd_token, LidarPointCloud)  # (D,N)
    xyz = pts[:3, :].T.astype(np.float32)
    return xyz


def load_radar_xyz_vxy(
    nusc,
    sample_token: str,
    sensor_channel: str = "RADAR_FRONT",
    max_range_m: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    returns radar_xyz (N,3), radar_vxy (N,2), radar_sd_token
    """
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"].get(sensor_channel, None)
    if sd_token is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), ""

    pts = _pc_to_ego(nusc, sd_token, RadarPointCloud)  # (D,N)

    xyz = pts[:3, :].T.astype(np.float32)
    x = pts[0, :]
    y = pts[1, :]

    vx = pts[6, :] if pts.shape[0] > 7 else np.zeros_like(x)
    vy = pts[7, :] if pts.shape[0] > 7 else np.zeros_like(x)

    if pts.shape[0] > 10:
        vx_c = pts[8, :]
        vy_c = pts[9, :]
        use_comp = ~np.isnan(vx_c) & ~np.isnan(vy_c)
        vx = np.where(use_comp, vx_c, vx)
        vy = np.where(use_comp, vy_c, vy)

    vxy = np.stack([vx, vy], axis=1).astype(np.float32)

    r = np.linalg.norm(np.stack([x, y], axis=1), axis=1)
    keep = r < float(max_range_m)
    return xyz[keep], vxy[keep], sd_token
