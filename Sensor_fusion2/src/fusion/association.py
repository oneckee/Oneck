# src/fusion/association.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def associate_nn(
    tracks_xy: np.ndarray,  # (T,2)
    dets_xy: np.ndarray,    # (D,2)
    gate_m: float = 3.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Simple association using distance gating.
    If scipy exists -> Hungarian; else greedy.
    """
    T = tracks_xy.shape[0]
    D = dets_xy.shape[0]
    if T == 0 or D == 0:
        return [], list(range(T)), list(range(D))

    # cost matrix (euclidean)
    cost = np.linalg.norm(tracks_xy[:, None, :] - dets_xy[None, :, :], axis=2)  # (T,D)

    # apply gating
    big = 1e9
    gated_cost = cost.copy()
    gated_cost[gated_cost > gate_m] = big

    matches: List[Tuple[int, int]] = []
    used_t = set()
    used_d = set()

    if _HAS_SCIPY:
        ti, di = linear_sum_assignment(gated_cost)
        for t, d in zip(ti.tolist(), di.tolist()):
            if gated_cost[t, d] >= big:
                continue
            matches.append((t, d))
            used_t.add(t)
            used_d.add(d)
    else:
        # greedy
        flat = [(gated_cost[t, d], t, d) for t in range(T) for d in range(D)]
        flat.sort(key=lambda x: x[0])
        for c, t, d in flat:
            if c >= big:
                break
            if t in used_t or d in used_d:
                continue
            matches.append((t, d))
            used_t.add(t)
            used_d.add(d)

    unmatched_tracks = [t for t in range(T) if t not in used_t]
    unmatched_dets = [d for d in range(D) if d not in used_d]
    return matches, unmatched_tracks, unmatched_dets
