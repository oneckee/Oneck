# src/fusion/radar_cluster.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple


def dbscan_2d(xy: np.ndarray, eps: float = 2.0, min_samples: int = 3) -> np.ndarray:
    """
    아주 가벼운 2D DBSCAN (N radar pts ~ 50이라 O(N^2) 충분)
    Returns labels: -1 noise, 0..K-1 clusters
    """
    N = xy.shape[0]
    if N == 0:
        return np.empty((0,), dtype=np.int32)

    # pairwise distances
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)  # (N,N)
    neighbors = [np.where(d[i] <= eps)[0].tolist() for i in range(N)]

    labels = np.full(N, -1, dtype=np.int32)
    visited = np.zeros(N, dtype=bool)

    cid = 0
    for i in range(N):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = neighbors[i]
        if len(nbrs) < min_samples:
            labels[i] = -1
            continue

        # expand cluster
        labels[i] = cid
        queue = list(nbrs)
        qi = 0
        while qi < len(queue):
            j = queue[qi]
            qi += 1
            if not visited[j]:
                visited[j] = True
                nbrs_j = neighbors[j]
                if len(nbrs_j) >= min_samples:
                    queue.extend([k for k in nbrs_j if k not in queue])
            if labels[j] == -1:
                labels[j] = cid
        cid += 1

    return labels


def radar_clusters_to_objects(
    radar_xy: np.ndarray,
    radar_vxy: np.ndarray,
    eps: float = 2.0,
    min_samples: int = 3,
    min_cluster_size: int = 4,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Returns list of (centroid_xy, mean_vxy, size)
    """
    if radar_xy.shape[0] == 0:
        return []

    labels = dbscan_2d(radar_xy, eps=eps, min_samples=min_samples)
    objs: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for cid in sorted(set(labels.tolist())):
        if cid < 0:
            continue
        idx = np.where(labels == cid)[0]
        if idx.size < min_cluster_size:
            continue
        cxy = radar_xy[idx].mean(axis=0)
        cv = radar_vxy[idx].mean(axis=0) if radar_vxy.shape[0] == radar_xy.shape[0] else np.zeros(2)
        objs.append((cxy, cv, int(idx.size)))
    return objs
