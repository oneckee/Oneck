import cv2
import numpy as np
import os

class BEVVisualizer: # 클래스명이 정확히 일치해야 합니다.
    def __init__(self, output_dir='./output/bev_frames', canvas_size=800, scale=10):
        self.output_dir = output_dir
        self.canvas_size = canvas_size
        self.scale = scale
        self.center = canvas_size // 2
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def render(self, tracks, frame_idx):
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        
        # 가이드라인(격자)
        for r in range(5, 50, 5):
            cv2.circle(canvas, (self.center, self.center), r * self.scale, (60, 60, 60), 1)

        # 자차 표시
        cv2.rectangle(canvas, (self.center-8, self.center-12), (self.center+8, self.center+12), (255, 255, 255), -1)

        for track in tracks:
            # track.ekf.x -> track.x 로 수정하여 AttributeError 방지
            x, y = track.x[0], track.x[1]
            
            img_x = int(self.center - (y * self.scale))
            img_y = int(self.center - (x * self.scale))

            if 0 <= img_x < self.canvas_size and 0 <= img_y < self.canvas_size:
                cv2.circle(canvas, (img_x, img_y), 6, (0, 255, 0), -1)
                cv2.putText(canvas, f"ID:{track.id} {track.label}", (img_x + 8, img_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        save_path = os.path.join(self.output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(save_path, canvas)