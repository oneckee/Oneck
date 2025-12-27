# src/fusion/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Optional
import numpy as np

Source = Literal["radar", "cam_gt"]


@dataclass
class Detection:
    x: float
    y: float
    vx: float
    vy: float
    score: float
    source: Source


@dataclass
class Track:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    P: np.ndarray  # (4,4)
    age: int = 0
    hits: int = 0
    misses: int = 0
    history: List[np.ndarray] = field(default_factory=list)

    def state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy], dtype=np.float64)
