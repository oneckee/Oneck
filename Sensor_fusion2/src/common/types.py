from dataclasses import dataclass, field
import numpy as np
from typing import Optional, List

@dataclass
class Detection3D:
    label: str
    translation: np.ndarray  # [x, y, z]
    size: np.ndarray         # [w, l, h]
    velocity: Optional[np.ndarray] = None  # [vx, vy] (Radar용)
    score: float = 0.0
    sensor_id: str = ""      # 어떤 센서에서 왔는지 (LIDAR, RADAR_FRONT 등)

@dataclass
class Track:
    track_id: int
    label: str
    state: np.ndarray        # [x, y, z, vx, vy, yaw, yaw_rate]
    covariance: np.ndarray
    age: int = 1
    hits: int = 1
    status: str = "tentative" # tentative, confirmed, deleted