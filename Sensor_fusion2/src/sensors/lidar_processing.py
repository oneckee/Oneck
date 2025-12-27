import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud

class LidarProcessor:
    def __init__(self, nusc):
        self.nusc = nusc

    def process_frame(self, sample_token):
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        data_path, _, _ = self.nusc.get_sample_data(lidar_token)
        
        # 라이다 데이터 로드 [x, y, z, intensity]
        lpc = LidarPointCloud.from_file(data_path)
        points = lpc.points.T 
        
        # 1. ROI 필터링 (자차 기준 전방 50m, 좌우 30m 등)
        mask = (np.abs(points[:, 0]) < 50) & (np.abs(points[:, 1]) < 30)
        points = points[mask]
        
        # 2. 지면 제거 (Ground Removal) - 간단한 Z축 필터링 예시
        ground_mask = points[:, 2] > -1.5 
        non_ground_points = points[ground_mask]
        
        return non_ground_points