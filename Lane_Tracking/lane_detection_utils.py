import cv2
import numpy as np




def process_image_for_lane_detection(image):
    """ 이미지 전처리: Grayscale -> Blur -> Canny Edge """

    # --- [새로 추가된 색상 필터링 로직] ---
    # 1. BGR -> HLS 변환 (HLS가 조명 변화에 강함)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
    # 2. 흰색 차선 필터링
    # H: 색상, L: 밝기, S: 채도
    lower_white = np.array([0, 200, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hls, lower_white, upper_white)
    
    # 3. 노란색 차선 필터링
    lower_yellow = np.array([15, 30, 115], dtype=np.uint8) # HLS에서 노란색 범위
    upper_yellow = np.array([35, 200, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)
    
    # 4. 흰색과 노란색 마스크 결합
    mask_lanes = cv2.bitwise_or(mask_white, mask_yellow)
    
# 4. 흰색과 노란색 마스크 결합
    mask_lanes = cv2.bitwise_or(mask_white, mask_yellow)
    
    # 5. 차선 마스크만 남기기
    masked_lanes_image = cv2.bitwise_and(image, image, mask=mask_lanes)
    # ----------------------------------------------------

    # ... (6, 7, 8 단계 코드 유지) ...
    
    

    # 5. 차선 마스크만 남기기
    masked_lanes_image = cv2.bitwise_and(image, image, mask=mask_lanes)
    # ----------------------------------------------------

    # 6. Grayscale 변환 (필터링된 이미지를 사용)
    gray = cv2.cvtColor(masked_lanes_image, cv2.COLOR_BGR2GRAY)
    
    # 7. Gaussian Blur 적용
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 8. Canny Edge Detection
    canny = cv2.Canny(blur, 50, 150)
    color_mask = cv2.cvtColor(mask_lanes, cv2.COLOR_GRAY2BGR) 
    
    return canny, masked_lanes_image, color_mask
    

def region_of_interest(image):
    """
    차선 감지 영역(ROI) 설정
    대시보드와 하늘 부분을 제외하도록 좌표를 수정
    """
    height = image.shape[0]
    width = image.shape[1]
    
    # [수정된 좌표]
    # 이전: (width * 0.45, height * 0.6)
    # 수정: 상단 높이를 이미지의 70% 지점(0.7)으로 내리고, 하단 너비를 중앙에 가깝게 조정
    
    polygon = np.array([
        # (좌측 하단), (우측 하단), (우측 상단), (좌측 상단)
        [(width , height ),  # 하단 우측     
         (width * 0 , height ),  # 하단 좌측
         (width * 0.4, height * 0.5),  # 상단 좌측
         (width * 0.6, height * 0.5)]  # 상단 우측
    ], np.int32)
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image, polygon


def make_coordinates(image, line_parameters):
    """ 기울기와 절편으로부터 실제 차선 좌표 계산 """
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    
    # 0으로 나누는 오류 방지 (수직선에 가까울 때)
    if slope == 0:
        return np.array([0, y1, 0, y2])
        
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    """ 좌/우측 차선으로 분류하고 평균하여 하나의 대표 차선 생성 """
    left_fit = [] 
    right_fit = [] 
    
    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        # 극단적인 기울기(수평선)를 필터링할 수 있음
        if abs(slope) < 0.1:
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
            
    # 좌/우측 대표선 계산
    left_fit_avg = np.average(left_fit, axis=0) if len(left_fit) > 0 else None
    right_fit_avg = np.average(right_fit, axis=0) if len(right_fit) > 0 else None

    left_line = make_coordinates(image, left_fit_avg) if left_fit_avg is not None else None
    right_line = make_coordinates(image, right_fit_avg) if right_fit_avg is not None else None
    
    return left_line, right_line

def display_lines(image, lines):
    """ 감지된 차선들을 이미지에 그리기 """
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None and len(line) == 4:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10) # 차선을 초록색으로 변경
    return line_image