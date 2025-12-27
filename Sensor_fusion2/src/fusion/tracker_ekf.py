# src/fusion/tracker.py
from __future__ import annotations
from typing import List

import numpy as np

from common.types import Detection, Track
from fusion.association import associate_nn


class CVKalmanTracker:
    """
    Constant Velocity Kalman Filter per track:
      state: [x, y, vx, vy]
    """

    def __init__(
        self,
        dt: float = 0.5,
        process_noise: float = 1.0,
        meas_noise_pos: float = 1.5,
        meas_noise_vel: float = 2.5,
        gate_m: float = 4.0,
        max_misses: int = 8,
        min_hits: int = 2,
    ):
        self.dt = float(dt)
        self.q = float(process_noise)
        self.r_pos = float(meas_noise_pos)
        self.r_vel = float(meas_noise_vel)
        self.gate_m = float(gate_m)
        self.max_misses = int(max_misses)
        self.min_hits = int(min_hits)

        self._next_id = 1
        self.tracks: List[Track] = []

    def _F(self) -> np.ndarray:
        dt = self.dt
        return np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

    def _Q(self) -> np.ndarray:
        # simple isotropic process noise
        q = self.q
        return np.eye(4, dtype=np.float64) * q

    def predict(self):
        F = self._F()
        Q = self._Q()
        for trk in self.tracks:
            x = trk.state()
            x = F @ x
            P = F @ trk.P @ F.T + Q
            trk.x, trk.y, trk.vx, trk.vy = x.tolist()
            trk.P = P
            trk.age += 1
            trk.history.append(np.array([trk.x, trk.y], dtype=np.float64))

    def _update_track(self, trk: Track, det: Detection):
        """
        Radar det has velocity; cam_gt det has only position (vx,vy=0 but we treat as unknown).
        We'll do:
          - if source == radar: update x,y,vx,vy
          - else: update x,y only
        """
        x = trk.state()
        P = trk.P

        if det.source == "radar":
            z = np.array([det.x, det.y, det.vx, det.vy], dtype=np.float64)
            H = np.eye(4, dtype=np.float64)
            R = np.diag([self.r_pos**2, self.r_pos**2, self.r_vel**2, self.r_vel**2])
        else:
            z = np.array([det.x, det.y], dtype=np.float64)
            H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]], dtype=np.float64)
            R = np.diag([self.r_pos**2, self.r_pos**2])

        y = z - (H @ x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x2 = x + K @ y
        P2 = (np.eye(4) - K @ H) @ P

        trk.x, trk.y, trk.vx, trk.vy = x2.tolist()
        trk.P = P2

        trk.hits += 1
        trk.misses = 0

    def _spawn(self, det: Detection):
        x = np.array([det.x, det.y, det.vx, det.vy], dtype=np.float64)
        P = np.eye(4, dtype=np.float64) * 10.0
        trk = Track(
            track_id=self._next_id,
            x=float(x[0]),
            y=float(x[1]),
            vx=float(x[2]),
            vy=float(x[3]),
            P=P,
            age=1,
            hits=1,
            misses=0,
        )
        trk.history.append(np.array([trk.x, trk.y], dtype=np.float64))
        self._next_id += 1
        self.tracks.append(trk)

    def step(self, detections: List[Detection]) -> List[Track]:
        """
        predict -> associate -> update -> spawn -> prune
        """
        self.predict()

        tracks_xy = np.array([[t.x, t.y] for t in self.tracks], dtype=np.float64)
        dets_xy = np.array([[d.x, d.y] for d in detections], dtype=np.float64)

        matches, un_trk, un_det = associate_nn(tracks_xy, dets_xy, gate_m=self.gate_m)

        # update matched
        for ti, di in matches:
            self._update_track(self.tracks[ti], detections[di])

        # mark unmatched tracks
        for ti in un_trk:
            self.tracks[ti].misses += 1

        # spawn unmatched dets
        for di in un_det:
            self._spawn(detections[di])

        # prune dead
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        # return confirmed tracks
        confirmed = [t for t in self.tracks if t.hits >= self.min_hits]
        return confirmed
