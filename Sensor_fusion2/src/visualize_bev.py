import cv2
import numpy as np
import os

class BEVVideoWriter:
    def __init__(self, output_path="output/track_result.mp4", fps=10, size=(1200, 800)):
        """
        최종 센서 퓨전 결과를 영상으로 저장하는 클래스
        """
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        self.size = size

    def combine_dashboard(self, cam_images, bev_canvas):
        """
        6개 카메라 이미지와 BEV 캔버스를 하나의 대시보드로 병합
        [ CAM_FL ][ CAM_F  ][ CAM_FR ]
        [ CAM_BL ][ BEV_MAP][ CAM_BR ]  <-- 이런 식의 배치
        """
        h, w = 200, 300 # 각 카메라 이미지 크기 조절
        
        # 카메라 이미지 리사이즈 및 배치
        imgs = {k: cv2.resize(v, (w, h)) for k, v in cam_images.items()}
        
        # 상단 열: Front Left, Front, Front Right
        top_row = np.hstack([imgs['CAM_FRONT_LEFT'], imgs['CAM_FRONT'], imgs['CAM_FRONT_RIGHT']])
        
        # 하단 열을 위한 BEV 리사이즈 (중앙에 배치하기 위해 크기 맞춤)
        bev_resized = cv2.resize(bev_canvas, (w, 400)) # BEV를 세로로 길게
        
        # 최종 대시보드 구성 (단순화된 예시 배치)
        # 실제로는 np.vstack과 np.hstack을 조합하여 1200x800 사이즈를 완성합니다.
        
        # 가이드용 검은 배경 생성
        dashboard = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        dashboard[:h, :w*3] = top_row
        dashboard[h:, w:w*2] = bev_resized # 중앙에 BEV 배치
        
        return dashboard

    def write_frame(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
        print(f"영상 저장 완료: {self.output_path}")

def create_bev_canvas(lidar_points, tracks, size=(800, 800)):
    """
    실시간 plt 화면이 아닌, OpenCV용 도화지에 BEV를 그리는 함수
    """
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    center = size[0] // 2
    scale = 10 # 1m당 10픽셀
    
    # LiDAR 점 그리기 (녹색 점)
    for p in lidar_points:
        px = int(center + p[0] * scale)
        py = int(center - p[1] * scale)
        if 0 <= px < size[0] and 0 <= py < size[1]:
            canvas[py, px] = [50, 50, 50]
            
    # Track 결과 그리기 (파란색 사각형 및 ID)
    for track in tracks:
        tx = int(center + track.state[0] * scale)
        ty = int(center - track.state[1] * scale)
        if 0 <= tx < size[0] and 0 <= ty < size[1]:
            cv2.circle(canvas, (tx, ty), 5, (255, 0, 0), -1)
            cv2.putText(canvas, f"ID:{track.track_id}", (tx+5, ty), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
    return canvas