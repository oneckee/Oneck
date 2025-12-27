import numpy as np

class TrackerEKF:
    def __init__(self, initial_state):
        """
        initial_state: [x, y, vx, vy] 등의 초기값
        """
        # 상태 벡터 x: [pos_x, pos_y, vel_x, vel_y]
        self.x = np.array(initial_state).reshape(-1, 1)
        
        # 공분산 행렬 P
        self.P = np.eye(4) * 1.0
        
        # 프로세스 노이즈 Q
        self.Q = np.eye(4) * 0.1
        
        # 상태 전이 행렬 F (등속 모델 예시)
        self.F = np.eye(4)

    def predict(self, ego_motion):
        """
        자차 움직임(ego_motion)을 반영하여 다음 상태 예측
        """
        # 1. 등속 모델 기반 예측
        self.x = self.F @ self.x
        
        # 2. 자차 움직임 보정 (Ego-motion compensation)
        # ego_motion['pos']와 ['rotation']을 사용하여 x, y 좌표 변환
        rot_matrix = ego_motion['rotation'].rotation_matrix[:2, :2]
        translation = ego_motion['pos'][:2].reshape(2, 1)
        
        # 위치 좌표 변환 적용
        self.x[:2] = rot_matrix @ self.x[:2] + translation
        
        # 공분산 업데이트
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """
        센서 측정값을 통한 상태 업데이트
        """
        # 관측 행렬 H (위치 x, y만 관측한다고 가정)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        z = np.array(measurement).reshape(2, 1)
        R = np.eye(2) * 0.5 # 측정 노이즈
        
        # 칼만 이득 계산 및 상태 업데이트
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P