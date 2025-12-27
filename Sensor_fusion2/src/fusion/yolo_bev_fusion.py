# src/fusion/yolo_bev_fusion.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False


@dataclass
class YoloDet:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    conf: float


def _get_T_ego_to_global(nusc, sd_token: str) -> np.ndarray:
    sd = nusc.get("sample_data", sd_token)
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return transform_matrix(ep["translation"], Quaternion(ep["rotation"]), inverse=False)


def _get_T_global_to_ego(nusc, sd_token: str) -> np.ndarray:
    sd = nusc.get("sample_data", sd_token)
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return transform_matrix(ep["translation"], Quaternion(ep["rotation"]), inverse=True)


def _get_T_ego_to_cam(nusc, cam_sd_token: str) -> np.ndarray:
    """
    cam calibrated_sensor: sensor->ego, so ego->cam = inverse
    """
    sd = nusc.get("sample_data", cam_sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    return transform_matrix(cs["translation"], Quaternion(cs["rotation"]), inverse=True)


def _transform_points(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    """
    pts_xyz: (N,3)
    """
    if pts_xyz.shape[0] == 0:
        return pts_xyz
    ones = np.ones((pts_xyz.shape[0], 1), dtype=np.float64)
    p = np.hstack([pts_xyz.astype(np.float64), ones])  # (N,4)
    out = (T @ p.T).T[:, :3]
    return out


def _project_cam(K: np.ndarray, pts_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    pts_cam: (N,3) in camera frame. returns uv (N,2) and depth z (N,)
    """
    z = pts_cam[:, 2]
    valid = z > 0.5
    pts = pts_cam[valid]
    zv = z[valid]
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    uvw = (K @ pts.T).T  # (N,3)
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, zv


def run_yolo_on_cam(
    img_bgr: np.ndarray,
    model: Any,
    conf_th: float = 0.5,
    iou_th: float = 0.6,
    allow_names: Tuple[str, ...] = ("car", "truck", "bus", "motorcycle", "bicycle", "person"),
) -> List[YoloDet]:
    """
    Ultralytics YOLOv8 결과를 bbox 리스트로 변환
    """
    if not _HAS_YOLO:
        raise RuntimeError("ultralytics YOLO not installed. pip install ultralytics")

    res = model.predict(img_bgr, conf=conf_th, iou=iou_th, verbose=False)[0]
    names = res.names  # dict id->name
    dets: List[YoloDet] = []
    if res.boxes is None:
        return dets

    for b in res.boxes:
        cls = int(b.cls.item())
        conf = float(b.conf.item())
        name = names.get(cls, str(cls))
        if name not in allow_names:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        dets.append(YoloDet(x1, y1, x2, y2, cls, conf))
    return dets


def yolo_to_bev_detections_via_lidar_radar(
    nusc,
    sample: Dict[str, Any],
    cam_channel: str,
    lidar_sd_token: str,
    lidar_xyz_ego: np.ndarray,   # in lidar-ego frame
    radar_sd_token: str,
    radar_xyz_ego: np.ndarray,   # in radar-ego frame
    radar_vxy: np.ndarray,
    yolo_model: Any,
    conf_th: float = 0.5,
    iou_th: float = 0.6,
    max_range_m: float = 60.0,
    subsample_lidar: int = 1,
) -> List[Tuple[float, float, float, float, float]]:
    """
    반환: [(x,y,vx,vy,score), ...] in CAMERA EGO frame (ego of camera timestamp)
    - 각 YOLO bbox 안에 들어오는 LiDAR/Radar 포인트를 projection으로 골라서,
      그 포인트들의 ego_cam 상에서 median x,y를 object 위치로 둠.
    """
    # cam sample_data token
    cam_sd_token = sample["data"].get(cam_channel, None)
    if cam_sd_token is None:
        return []

    # image load
    img_path = nusc.get_sample_data_path(cam_sd_token)
    img = cv2.imread(img_path)  # BGR
    if img is None:
        return []

    # calib
    cam_sd = nusc.get("sample_data", cam_sd_token)
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    K = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)

    # YOLO
    yolo_dets = run_yolo_on_cam(img, yolo_model, conf_th=conf_th, iou_th=iou_th)

    # --- transform LiDAR/Radar points into ego_cam, then to cam for projection
    # lidar ego -> global -> ego_cam
    T_lidar_ego_to_global = _get_T_ego_to_global(nusc, lidar_sd_token)
    T_global_to_ego_cam = _get_T_global_to_ego(nusc, cam_sd_token)

    # radar ego -> global -> ego_cam
    T_radar_ego_to_global = _get_T_ego_to_global(nusc, radar_sd_token)

    # ego_cam -> cam
    T_ego_cam_to_cam = _get_T_ego_to_cam(nusc, cam_sd_token)

    # lidar
    lidar_xyz = lidar_xyz_ego
    if lidar_xyz.shape[0] > 0 and subsample_lidar > 1:
        lidar_xyz = lidar_xyz[::subsample_lidar]

    lidar_xyz_global = _transform_points(T_lidar_ego_to_global, lidar_xyz)
    lidar_xyz_ego_cam = _transform_points(T_global_to_ego_cam, lidar_xyz_global)
    lidar_xyz_cam = _transform_points(T_ego_cam_to_cam, lidar_xyz_ego_cam)

    uv_l, depth_l = _project_cam(K, lidar_xyz_cam)

    # radar
    radar_xyz_global = _transform_points(T_radar_ego_to_global, radar_xyz_ego)
    radar_xyz_ego_cam = _transform_points(T_global_to_ego_cam, radar_xyz_global)
    radar_xyz_cam = _transform_points(T_ego_cam_to_cam, radar_xyz_ego_cam)

    uv_r, depth_r = _project_cam(K, radar_xyz_cam)

    # NOTE: uv arrays are for valid z>0.5 only; to keep indices aligned, we re-filter similarly
    # We'll rebuild valid masks to index original points in ego_cam:
    def valid_cam(pts_cam: np.ndarray) -> np.ndarray:
        return pts_cam[:, 2] > 0.5

    vmask_l = valid_cam(lidar_xyz_cam)
    vmask_r = valid_cam(radar_xyz_cam)

    lidar_ego_cam_valid = lidar_xyz_ego_cam[vmask_l]
    radar_ego_cam_valid = radar_xyz_ego_cam[vmask_r]
    radar_vxy_valid = radar_vxy[vmask_r] if radar_vxy.shape[0] == radar_xyz_ego.shape[0] else np.zeros((radar_ego_cam_valid.shape[0], 2))

    out: List[Tuple[float, float, float, float, float]] = []

    for d in yolo_dets:
        x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2

        # 1) LiDAR points inside bbox
        if uv_l.shape[0] > 0:
            inside_l = (uv_l[:, 0] >= x1) & (uv_l[:, 0] <= x2) & (uv_l[:, 1] >= y1) & (uv_l[:, 1] <= y2)
            pts_l = lidar_ego_cam_valid[inside_l]
        else:
            pts_l = np.zeros((0, 3), dtype=np.float64)

        # 2) Radar points inside bbox (fallback or velocity)
        if uv_r.shape[0] > 0:
            inside_r = (uv_r[:, 0] >= x1) & (uv_r[:, 0] <= x2) & (uv_r[:, 1] >= y1) & (uv_r[:, 1] <= y2)
            pts_r = radar_ego_cam_valid[inside_r]
            v_r = radar_vxy_valid[inside_r]
        else:
            pts_r = np.zeros((0, 3), dtype=np.float64)
            v_r = np.zeros((0, 2), dtype=np.float64)

        # choose position source
        if pts_l.shape[0] >= 2:
            x_med = float(np.median(pts_l[:, 0]))
            y_med = float(np.median(pts_l[:, 1]))
            vx, vy = 0.0, 0.0
        elif pts_r.shape[0] >= 1:
            x_med = float(np.median(pts_r[:, 0]))
            y_med = float(np.median(pts_r[:, 1]))
            vx = float(np.mean(v_r[:, 0])) if v_r.shape[0] else 0.0
            vy = float(np.mean(v_r[:, 1])) if v_r.shape[0] else 0.0
        else:
            continue

        r = (x_med * x_med + y_med * y_med) ** 0.5
        if r > max_range_m:
            continue

        out.append((x_med, y_med, vx, vy, float(d.conf)))

    return out


def build_yolo_model(weights: str = "yolov8n.pt"):
    if not _HAS_YOLO:
        raise RuntimeError("ultralytics YOLO not installed. pip install ultralytics")
    return YOLO(weights)
