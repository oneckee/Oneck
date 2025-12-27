import numpy as np
from scipy.optimize import linear_sum_assignment

class DataAssociation:
    def __init__(self, gate_threshold=5.0):
        self.gate_threshold = gate_threshold

    def compute_distance_matrix(self, tracks, detections):
        """트랙과 검출값 사이의 거리 행렬 계산"""
        num_tracks = len(tracks)
        num_dets = len(detections)
        dist_matrix = np.zeros((num_tracks, num_dets))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # 단순 유클리드 거리 (실차에선 Mahalanobis 거리 권장)
                dist = np.linalg.norm(track.state[:2] - det.translation[:2])
                dist_matrix[i, j] = dist
        return dist_matrix

    def associate(self, tracks, detections):
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        dist_matrix = self.compute_distance_matrix(tracks, detections)
        # 헝가리안 알고리즘을 이용한 최적 매칭
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] < self.gate_threshold:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)

        return matches, unmatched_tracks, unmatched_dets