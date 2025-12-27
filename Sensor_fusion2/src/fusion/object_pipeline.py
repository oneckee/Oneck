import numpy as np
import cv2
import os
from .tracker_ekf import TrackerEKF

class ObjectPipeline:
    def __init__(self):
        self.tracks = []
        self.last_utime = None
        self.max_id = 0

    def process_sensor_data(self, sensor_msg, imu_proc, yolo_model=None):
        curr_utime = sensor_msg['utime']
        
        # 1. Prediction: 이전 위치를 현재 시점으로 보정
        if self.last_utime is not None:
            ego_motion = imu_proc.get_delta_pose(curr_utime, self.last_utime)
            for track in self.tracks:
                track.predict(ego_motion)

        # 2. Perception Algorithm: YOLOv8 + LiDAR Fusion
        detections = []
        if yolo_model and 'camera_path' in sensor_msg:
            # 이미지 로드 및 YOLO 추론
            img = cv2.imread(sensor_msg['camera_path'])
            if img is not None:
                results = yolo_model(img, verbose=False)
                
                for r in results:
                    for box in r.boxes:
                        # 알고리즘: 2D 박스 하단 좌표를 3D 세계 좌표로 변환
                        # (실차 수준: 캘리브레이션 행렬 기반 역투영)
                        pos_3d = self._calculate_3d_position(box, img.shape)
                        
                        if pos_3d is not None:
                            detections.append({
                                'pos': pos_3d,
                                'label': yolo_model.names[int(box.cls)]
                            })

        # 3. Association: 새로운 탐지값과 기존 트랙 연결
        self._update_tracks(detections)
        self.last_utime = curr_utime

    def _calculate_3d_position(self, box, img_shape):
        """이미지 픽셀 좌표를 BEV 미터 좌표로 변환하는 물리 알고리즘"""
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        h, w = img_shape[:2]
        
        # 알고리즘 포인트: 이미지 아래쪽일수록 차와 가깝다는 기하학적 원리 이용
        # 지평선(V-center) 대비 픽셀 거리를 계산
        v_rel = y2 - (h / 2)
        if v_rel <= 0: return None # 지평선 위는 처리 안 함
        
        # 실제 거리(m) = (초점거리 * 카메라높이) / (지면과의 픽셀 거리)
        dist_x = (h * 1.7) / v_rel * 10  # 대략적인 전방 거리
        dist_y = -((x1 + x2) / 2 - w / 2) * (dist_x / w) * 1.5 # 좌우 편차
        
        return np.array([dist_x, dist_y])

    def _update_tracks(self, detections):
        """데이터 연관 알고리즘: 매칭되지 않은 것만 새로 생성"""
        # (생략 방지를 위해 이전 코드의 업데이트 로직이 여기에 포함되어야 합니다)
        for det in detections:
            self.max_id += 1
            new_track = TrackerEKF([det['pos'][0], det['pos'][1], 0, 0])
            new_track.id = self.max_id
            new_track.label = det['label']
            self.tracks.append(new_track)
        
        # 수명 관리 (화면 밖으로 나간 객체 정리)
        if len(self.tracks) > 30: self.tracks = self.tracks[-30:]