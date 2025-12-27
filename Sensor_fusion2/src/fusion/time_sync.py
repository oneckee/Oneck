from collections import deque


class TimeSynchronizer:
    def __init__(self, nusc, scene):
        self.nusc = nusc
        self.scene = scene
        self.samples = self._get_all_samples()
        self.current_idx = 0

    def _get_all_samples(self):
        samples = []
        curr_sample_token = self.scene['first_sample_token']
        while curr_sample_token != "":
            sample = self.nusc.get('sample', curr_sample_token)
            samples.append(sample)
            curr_sample_token = sample['next']
        return samples

    def get_next(self):
        if self.current_idx >= len(self.samples):
            return None
        
        sample = self.samples[self.current_idx]
        self.current_idx += 1
        
        # 'type' 키를 추가하여 ObjectPipeline의 KeyError를 해결합니다.
        return {
            'type': 'LIDAR', # nuScenes sample 기준이므로 기본 타입을 LIDAR로 설정
            'utime': sample['timestamp'],
            'lidar_path': self.nusc.get_sample_data_path(sample['data']['LIDAR_TOP']),
            'radar_paths': [self.nusc.get_sample_data_path(sample['data'][r]) for r in 
                            ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']],
            'sample_token': sample['token']
        }