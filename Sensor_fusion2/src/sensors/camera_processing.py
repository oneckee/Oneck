import cv2
import os

class CameraProcessor:
    def __init__(self, nusc, dataset_root):
        self.nusc = nusc
        self.root = dataset_root
        self.sensor_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
        ]

    def get_all_images(self, sample_token):
        """한 프레임에 해당하는 6대 카메라 이미지 로드"""
        sample = self.nusc.get('sample', sample_token)
        image_data = {}

        for sensor in self.sensor_names:
            sd_token = sample['data'][sensor]
            data_path, _, _ = self.nusc.get_sample_data(sd_token)
            full_path = os.path.join(self.root, data_path)
            
            img = cv2.imread(full_path)
            image_data[sensor] = img
            
        return image_data

    def detect_2d_batch(self, image_data, model):
        """6대 카메라 이미지에 대해 YOLO 등 AI 모델 배치 추론"""
        results = {}
        for sensor, img in image_data.items():
            # 실제 모델 추론 코드 (예: YOLOv8)
            # result = model.predict(img, conf=0.25)
            results[sensor] = [] # 검출 결과 저장
        return results