import os
import sys
from nuscenes.nuscenes import NuScenes
from sensors.imu_processing import IMUProcessor
from fusion.object_pipeline import ObjectPipeline
from visualization.bev_realtime import BEVVisualizer
from fusion.time_sync import TimeSynchronizer

def run_fusion_system():
    # 1. 경로 설정 (절대 경로)
    base_path = '/Users/oneck/Desktop/Sensor_Fusion2'
    dataroot = os.path.join(base_path, 'data/nuscenes')
    
    # 2. nuScenes 데이터셋 로드
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
    scene = nusc.scene[0] # scene-0061 등 분석할 씬 선택
    
    # 3. 각 모듈 초기화
    # TimeSynchronizer가 씬의 샘플들을 정상적으로 가져오도록 설정
    time_sync = TimeSynchronizer(nusc, scene)
    imu_proc = IMUProcessor(dataroot, scene['name'])
    pipeline = ObjectPipeline()
    
    # 시각화 결과 저장 경로 설정
    output_path = os.path.join(base_path, 'output/bev_frames')
    visualizer = BEVVisualizer(output_dir=output_path)

    print(f"\n🚀 {scene['name']} 분석 및 이미지 저장 시작...")
    print(f"📍 저장 위치: {output_path}")

    frame_idx = 0
    try:
        while True:
            # 다음 센서 데이터 가져오기 (utime, type 등이 포함된 msg)
            msg = time_sync.get_next()
            if msg is None:
                break
            
            # 4. 퓨전 파이프라인 처리
            pipeline.process_sensor_data(msg, imu_proc)
            
            # 5. 결과 시각화 및 이미지 파일 저장
            # pipeline.tracks 정보를 바탕으로 BEV 이미지를 생성합니다.
            visualizer.render(pipeline.tracks, frame_idx)
            
            # 터미널에 실시간 진행률 표시
            sys.stdout.write(f"\r처리 중: [Frame {frame_idx:03d}] ")
            sys.stdout.flush()
            
            frame_idx += 1

    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n✅ 분석 완료! 총 {frame_idx}개의 이미지가 저장되었습니다.")

if __name__ == "__main__":
    run_fusion_system()