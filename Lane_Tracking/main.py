import cv2
import numpy as np
import main_lane_tracker # 이전에 작성한 로직 파일 임포트

# --- 설정 ---
# 0: 기본 웹캠 사용, 'video.mp4' 등 파일 경로도 사용 가능
VIDEO_SOURCE = 'Video_Driving.mp4'
WINDOW_NAME = "Real-time Lane Tracking"
# ---

def main_loop():
    """
    비디오 스트림을 열고 각 프레임에 대해 차선 추적 알고리즘을 실행하는 메인 루프
    """
    
    # 비디오 캡처 객체 생성
   
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        return

    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            
            
            if not ret:
                print("End of video stream or failed to read frame.")
                break
                
            resized_frame = cv2.resize(frame, (640, 480))
            # 이미지 처리 및 조향 각도 계산
            # process_frame_for_steering 함수는 main_lane_tracker 모듈에 정의되어 있음
            processed_image, angle, offset, masked_lanes_image, color_mask= main_lane_tracker.process_frame_for_steering(resized_frame)
            
            # --- 결과 정보 화면 표시 ---
            # 조향 각도 텍스트 표시
            info_text = f"Steering Angle: {angle:.2f} | Offset: {offset:.2f} px"
            cv2.putText(processed_image, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 윈도우에 결과 표시
            cv2.imshow(WINDOW_NAME, processed_image)

            # 💡 디버깅 창 추가
            cv2.imshow("2. Color Filter Result (Masked Image)", masked_lanes_image)
            cv2.imshow("3. Color Mask (Black=Blocked)", color_mask)
            
            # 'q' 키를 누르면 루프 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        
    finally:
        # 작업이 끝나면 캡처 객체와 윈도우 해제
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()