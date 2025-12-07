import cv2
import numpy as np
import lane_detection_utils # 위에서 만든 유틸리티 모듈 임포트

def calculate_steering_angle(image, left_line, right_line):
    """
    감지된 차선을 기반으로 차량의 조향 각도 계산 및 오프셋 출력
    """
    if left_line is None or right_line is None:
        return 0.0, 0.0 # 조향 각도, 오프셋
    
    # 차선의 하단 x 좌표 평균 (차량의 현재 위치)
    x1_left, y1_left, x2_left, y2_left = left_line
    x1_right, y1_right, x2_right, y2_right = right_line
    
    # 안전하게 하단 y좌표 기준점을 y1_left와 y1_right의 평균으로 설정
    # (일반적으로 y1은 이미지의 최대 높이(하단)과 같음)
    
    # 차선의 하단 중간 지점 (차량이 따라가야 할 목표 X 좌표)
    lane_mid_x = (x1_left + x1_right) / 2
    
    # 이미지 중앙 X 좌표
    image_center_x = image.shape[1] / 2
    
    # 차량이 차선 중앙에서 얼마나 벗어났는지 (픽셀 오프셋)
    offset_x = lane_mid_x - image_center_x
    
    # 픽셀 오프셋을 조향 각도로 변환 (간단한 비례 제어)
    # 0.01은 조향 감도(Gain)로 튜닝이 필요
    steering_angle = offset_x * 0.01 
    
    return steering_angle, offset_x

def process_frame_for_steering(frame):
    """ 단일 프레임에서 차선 감지 및 조향 각도 계산까지 전체 파이프라인 실행 """
    
    # 1. 전처리 및 ROI
    canny_image, masked_lanes_image, color_mask = lane_detection_utils.process_image_for_lane_detection(frame)
    cropped_image, roi_polygon = lane_detection_utils.region_of_interest(canny_image)
    
    # 2. Hough 변환으로 직선 감지
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    # 3. 차선 평균화
    left_line, right_line = lane_detection_utils.average_slope_intercept(frame, lines)
    
    # 4. 조향 각도 계산
    steering_angle, offset = calculate_steering_angle(frame, left_line, right_line)
    
    # 5. 시각화 (좌/우측 차선 그리기)
    processed_lines = []
    if left_line is not None: processed_lines.append(left_line)
    if right_line is not None: processed_lines.append(right_line)

    line_image = lane_detection_utils.display_lines(frame, np.array(processed_lines))
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # 💡 경계선 시각화 코드 추가 (roi_polygon은 3차원 배열임)
    if roi_polygon is not None:
        # cv2.polylines: 다각형의 경계선만 그리는 함수
        # 닫힌 다각형 (isClosed=True), 색상=(0, 0, 255)는 빨간색, 2는 선 굵기
        # roi_polygon은 (1, N, 2) 형태이므로 그대로 전달
        cv2.polylines(combo_image, roi_polygon, isClosed=True, color=(0, 0, 255), thickness=2)

    # 중앙 오프셋 시각화 (빨간 점)
    center_x = int(frame.shape[1] / 2)
    cv2.circle(combo_image, (center_x, frame.shape[0]), 10, (0, 0, 255), -1) # 이미지 중앙
    if left_line is not None and right_line is not None:
        target_x = int((left_line[0] + right_line[0]) / 2)
        cv2.circle(combo_image, (target_x, frame.shape[0]), 10, (255, 0, 0), -1) # 차선 중앙 (목표점)

    return combo_image, steering_angle, offset, masked_lanes_image, color_mask

# --- 메인 실행 ---
if __name__ == "__main__":
    # 테스트 이미지 로드 (실제 경로로 수정 필요)
    try:
        image = cv2.imread('road_image.jpg') 
        if image is None:
             raise FileNotFoundError
    except FileNotFoundError:
        print("Error: road_image.jpg 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()
        
    final_image, angle, offset = process_frame_for_steering(image)
    
    print(f"Calculated Steering Angle: {angle:.2f} (Offset: {offset:.2f} pixels)")

    cv2.imshow("Lane Tracking Result", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()