# src/fusion/object_pipeline.py
from __future__ import annotations

from fusion.radar_cluster import radar_clusters_to_objects
from fusion.yolo_bev_fusion import build_yolo_model, yolo_to_bev_detections_via_lidar_radar
from sensors.nusc_loaders import load_lidar_xyz, load_radar_xyz_vxy
from common.types import Detection  # 너 프로젝트 기준


from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
from common.types import Detection, Track

from sensors.nusc_loaders import load_lidar_xy, load_radar_xy_vxy
from fusion.tracker_ekf import CVKalmanTracker



@dataclass
class PipelineConfig:
    # radar channel in nuScenes mini commonly:
    radar_channel: str = "RADAR_FRONT"
    lidar_channel: str = "LIDAR_TOP"

    # tracker params
    dt: float = 0.5
    gate_m: float = 5.0
    max_misses: int = 8
    min_hits: int = 2


class ObjectPipeline:
    """
    Minimal "실차 느낌" 파이프라인의 뼈대:
      - lidar/radar 로드(ego)
      - cam은 일단 GT annotation 중심점을 detection으로 사용 (YOLO는 다음 단계)
      - radar det는 vx/vy 포함, cam det는 pos only
      - tracker는 CV-KF로 track 유지
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.tracker = CVKalmanTracker(
            dt=cfg.dt,
            gate_m=cfg.gate_m,
            max_misses=cfg.max_misses,
            min_hits=cfg.min_hits,
        )

        self.yolo = build_yolo_model("yolov8n.pt")  # 필요하면 yolov8s.pt 등으로 변경
        self.cam_channel = "CAM_FRONT"              # 시작은 front로 (6캠 dedup은 다음 단계)


    def _cam_gt_detections(self, nusc, sample_token: str) -> List[Detection]:
        """
        nuScenes annotation(box) center를 ego frame으로 변환하는 대신,
        여기서는 'global' -> ego 변환을 하지 않고, 간단히:
          - LIDAR_TOP sample_data 기준 ego pose에 맞춰 box center를 ego로 변환
        """
        sample = nusc.get("sample", sample_token)
        ann_tokens = sample.get("anns", [])
        if not ann_tokens:
            return []

        # 기준 ego pose: LIDAR_TOP의 ego_pose
        sd_token = sample["data"].get(self.cfg.lidar_channel, None)
        if sd_token is None:
            return []

        sd = nusc.get("sample_data", sd_token)
        ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
        # ego global pose
        ego_t = np.array(ego_pose["translation"], dtype=np.float64)
        # quaternion (w,x,y,z in pyquaternion)
        from pyquaternion import Quaternion
        q = Quaternion(ego_pose["rotation"])
        R_g2e = q.rotation_matrix.T  # global->ego

        dets: List[Detection] = []
        for tok in ann_tokens:
            ann = nusc.get("sample_annotation", tok)
            c_g = np.array(ann["translation"], dtype=np.float64)  # global center

            # global -> ego
            c_e = R_g2e @ (c_g - ego_t)
            x, y = float(c_e[0]), float(c_e[1])

            dets.append(
                Detection(
                    x=x, y=y,
                    vx=0.0, vy=0.0,
                    score=0.7,
                    source="cam_gt",
                )
            )
        return dets

    def _radar_detections(self, radar_xy: np.ndarray, radar_vxy: np.ndarray) -> List[Detection]:
        dets: List[Detection] = []
        if radar_xy.shape[0] == 0:
            return dets
        for (x, y), (vx, vy) in zip(radar_xy.tolist(), radar_vxy.tolist()):
            dets.append(
                Detection(
                    x=float(x), y=float(y),
                    vx=float(vx), vy=float(vy),
                    score=0.9,
                    source="radar",
                )
            )
        return dets

    def process(self, nusc, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_token = sample["token"]

        # 1) load sensors in ego frame
        lidar_xy = load_lidar_xy(nusc, sample_token, sensor_channel=self.cfg.lidar_channel)

        radar_xy, radar_vxy = load_radar_xy_vxy(
            nusc, sample_token, sensor_channel=self.cfg.radar_channel, max_range_m=80.0
        )

        # 2) create detections
        cam_dets = self._cam_gt_detections(nusc, sample_token)
        radar_dets = self._radar_detections(radar_xy, radar_vxy)

        # 3) fusion detections (간단히 radar+cam을 한 리스트로)
        # detections = radar_dets + cam_dets

        # # 4) tracking
        # tracks = self.tracker.step(detections)

        sample_token = sample["token"]

   

        # ----------------------------------------------------
        # A) 센서 로드: "여기까지"가 A (B/C/D는 이 아래!)
        # ----------------------------------------------------
        lidar_xyz = load_lidar_xyz(nusc, sample_token, sensor_channel=self.cfg.lidar_channel)

        radar_xyz, radar_vxy, radar_sd_token = load_radar_xyz_vxy(
            nusc, sample_token, sensor_channel=self.cfg.radar_channel, max_range_m=80.0
        )

        # YOLO→BEV 융합에 lidar sample_data token 필요
        lidar_sd_token = sample["data"][self.cfg.lidar_channel]

        # ----------------------------------------------------
        # B) ✅ 여기! Radar point → 클러스터링 → "객체 det" 만들기
        #    (반드시 process() 안, A 아래에 들어가야 함)
        # ----------------------------------------------------
        radar_xy = radar_xyz[:, :2] if radar_xyz.shape[0] else np.zeros((0, 2), np.float32)

        radar_objs = radar_clusters_to_objects(
            radar_xy, radar_vxy,
            eps=3.0, min_samples=2, min_cluster_size=3
        )

        radar_obj_dets = [
            Detection(
                x=float(cxy[0]), y=float(cxy[1]),
                vx=float(cv[0]), vy=float(cv[1]),
                score=min(1.0, sz / 10.0),
                source="radar",
            )
            for (cxy, cv, sz) in radar_objs
        ]

        # ----------------------------------------------------
        # C) ✅ 여기! YOLO 2D bbox → LiDAR/Radar로 거리 붙여서 BEV det 생성
        #    (B 아래에 이어서)
        # ----------------------------------------------------
        yolo_bev = yolo_to_bev_detections_via_lidar_radar(
            nusc=nusc,
            sample=sample,
            cam_channel=self.cam_channel,   # 예: "CAM_FRONT"
            lidar_sd_token=lidar_sd_token,
            lidar_xyz_ego=lidar_xyz,
            radar_sd_token=radar_sd_token,
            radar_xyz_ego=radar_xyz,
            radar_vxy=radar_vxy,
            yolo_model=self.yolo,           # __init__에서 1번 로드한 모델
            conf_th=0.55,
            iou_th=0.65,
            max_range_m=60.0,
            subsample_lidar=6,
        )

        cam_bev_dets = [
            Detection(x=x, y=y, vx=vx, vy=vy, score=s, source="cam_yolo")
            for (x, y, vx, vy, s) in yolo_bev
        ]

        # ----------------------------------------------------
        # D) ✅ 여기! Tracker 입력은 "객체 det만" (raw radar point 넣지 말기)
        #    (C 아래에 이어서)
        # ----------------------------------------------------
        detections = radar_obj_dets + cam_bev_dets
        tracks = self.tracker.step(detections)

        
        return {
            "sample_token": sample_token,
            "lidar_xy": lidar_xy,
            "radar_xy": radar_xy,
            "radar_vxy": radar_vxy,
            "cam_dets": cam_dets,      # GT 기반
            "detections": detections,  # all
            "tracks": tracks,
        }
