import numpy as np
from pyquaternion import Quaternion

class SensorCalibrator:
    def __init__(self, nusc, sample_token):
        """13개 센서의 변환 행렬을 미리 계산하여 보관하는 관리자입니다. 📂"""
        self.tf_cache = {}
        sample = nusc.get('sample', sample_token)
        
        # nuScenes의 모든 센서 데이터를 순회하며 변환 행렬 생성
        for sensor, sd_token in sample['data'].items():
            sd = nusc.get('sample_data', sd_token)
            cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            
            # 센서 좌표계 -> 자차(Ego) 좌표계 변환 행렬
            self.tf_cache[sensor] = self.make_tf_matrix(cs['translation'], cs['rotation'])

    def make_tf_matrix(self, translation, rotation):
        """이동(T)과 회전(R) 데이터를 하나의 4x4 변환 행렬로 만듭니다. 📏"""
        tm = np.eye(4)
        tm[:3, :3] = Quaternion(rotation).rotation_matrix
        tm[:3, 3] = translation
        return tm

    def get_tf(self, sensor_name):
        """특정 센서의 변환 행렬을 반환합니다."""
        return self.tf_cache.get(sensor_name)

    def transform_points(self, points, sensor_name):
        """특정 센서의 점들을 자차(Ego) 좌표계로 변환합니다. 📍"""
        tf = self.get_tf(sensor_name)
        if tf is None: return points
        
        # Homogeneous coordinates 변환 (N, 3) -> (N, 4)
        points_h = np.column_stack((points, np.ones(len(points))))
        transformed_h = (tf @ points_h.T).T
        return transformed_h[:, :3]