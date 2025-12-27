import json
import os
import numpy as np
from pyquaternion import Quaternion

class IMUProcessor:
    def __init__(self, dataset_path, scene_name):
        """
        nuScenes can_bus 데이터를 로드하여 자차의 위치/회전 상태를 관리합니다. 📂
        dataset_path: 데이터셋 루트 경로 (예: /Users/.../data/nuscenes)
        scene_name: 씬 이름 (예: 'scene-0001')
        """
        self.can_bus_path = os.path.join(dataset_path, 'can_bus')
        
        # 파일 이름 규칙에 맞춰 데이터를 로드합니다.
        self.imu_data = self._load_json(f"{scene_name}_ms_imu.json")
        self.pose_data = self._load_json(f"{scene_name}_pose.json")

        # 데이터 로드 확인 (empty sequence 에러 방지) 🛡️
        if not self.pose_data:
            raise FileNotFoundError(f"❌ {scene_name}_pose.json 데이터를 찾을 수 없거나 비어 있습니다. 경로를 확인하세요: {self.can_bus_path}")

    def _load_json(self, filename):
        path = os.path.join(self.can_bus_path, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return []

    def get_ego_pose_at_time(self, utime):
        """특정 시점(utime)의 자차 Global Pose를 반환합니다. 📍"""
        # 타임스탬프가 가장 가까운 데이터를 찾습니다.
        closest_pose = min(self.pose_data, key=lambda x: abs(x['utime'] - utime))
        
        return {
            'translation': np.array(closest_pose['pos']),
            'rotation': Quaternion(closest_pose['orientation']),
            'vel': np.array(closest_pose['vel']),
            'rotation_rate': np.array(closest_pose['rotation_rate'])
        }

    def get_delta_pose(self, curr_utime, prev_utime):
        """
        두 시점 사이의 상대적 변환(Delta Pose)을 계산합니다. 🚗
        이 값은 ObjectPipeline에서 객체의 위치를 보정하는 데 사용됩니다.
        """
        curr_p = self.get_ego_pose_at_time(curr_utime)
        prev_p = self.get_ego_pose_at_time(prev_utime)

        # 1. 상대 회전 계산 (Relative Rotation)
        # 현재 회전에서 이전 회전의 역행렬을 곱해 변화량을 구합니다.
        delta_rotation = curr_p['rotation'] * prev_p['rotation'].inverse
        
        # 2. 상대 이동 계산 (Relative Translation)
        # 전역 좌표계 이동량을 구한 뒤, 이전 시점의 차량 좌표계 기준으로 회전시킵니다.
        global_delta_pos = curr_p['translation'] - prev_p['translation']
        relative_pos = prev_p['rotation'].inverse.rotate(global_delta_pos)

        return {
            'pos': relative_pos,           # [dx, dy, dz]
            'rotation': delta_rotation,     # Quaternion
            'dt': (curr_utime - prev_utime) / 1e6  # 시간 차이(초)
        }

    def compensate_ego_motion(self, points, curr_utime, prev_utime):
        """자차 이동에 따른 점군 데이터 보정 로직 (좌표 변환 활용) 📐"""
        delta = self.get_delta_pose(curr_utime, prev_utime)
        
        # 점군(points)의 각 점에 대해 회전과 이동을 적용합니다.
        # R * p + t 형태의 변환입니다.
        rotated_points = np.array([delta['rotation'].rotate(p) for p in points])
        compensated_points = rotated_points + delta['pos']
        
        return compensated_points