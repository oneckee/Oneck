import numpy as np
import os
from nuscenes.utils.data_classes import RadarPointCloud

class RadarProcessor:
    def __init__(self, nusc):
        self.nusc = nusc
        self.sensor_names = [
            'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 
            'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'
        ]

    def get_merged_radar_points(self, sample_token):
        """5대 레이다 데이터를 Ego 좌표계로 통합"""
        sample = self.nusc.get('sample', sample_token)
        all_points = []
        
        for sensor in self.sensor_names:
            sd_token = sample['data'][sensor]
            data_path, boxes, _ = self.nusc.get_sample_data(sd_token)
            
            # PCD 파일 로드 (nuScenes 전용 로더 사용)
            rpc = RadarPointCloud.from_file(data_path)
            
            # 레이다 포인트의 속도(vx, vy) 및 유효성 필터링 (RCS 기반)
            # nuScenes Radar 데이터는 18개의 필드를 가짐 (x, y, z, rcs, vx, vy 등)
            points = rpc.points.T # [N, 18]
            
            # 유효한 포인트만 선택 (예: RCS > 0)
            valid_mask = points[:, 3] > 0 
            valid_points = points[valid_mask]
            
            # 센서 좌표 -> Ego 좌표로 변환 필요 (Calibration 활용)
            # 여기서는 편의상 포인트 리스트에 추가하는 구조만 구현
            all_points.append(valid_points)
            
        return np.vstack(all_points) if all_points else np.array([])