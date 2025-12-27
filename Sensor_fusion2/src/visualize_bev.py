# src/visualize_bev.py
from __future__ import annotations

from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt


def _setup_ax(ax, lim=60.0):
    ax.set_aspect("equal", "box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_xlabel("x (forward, m)")
    ax.set_ylabel("y (left, m)")


def render_bev_frame(frame: Dict[str, Any], save_path: str, dpi: int = 130, lim: float = 60.0):
    """
    Render:
      - lidar points (gray)
      - radar points (green)
      - cam_gt detections (orange x)
      - tracks (red) with velocity arrow
    """
    lidar_xy: np.ndarray = frame["lidar_xy"]
    radar_xy: np.ndarray = frame["radar_xy"]
    radar_vxy: np.ndarray = frame["radar_vxy"]
    cam_dets = frame["cam_dets"]
    tracks = frame["tracks"]

    fig = plt.figure(figsize=(6.2, 6.2), dpi=dpi)
    ax = fig.add_subplot(111)
    _setup_ax(ax, lim=lim)

    # ego marker
    ax.scatter([0], [0], s=40, marker="s")

    # lidar
    if lidar_xy.shape[0] > 0:
        ax.scatter(lidar_xy[:, 0], lidar_xy[:, 1], s=1)

    # radar points + velocity arrows (downsample to avoid clutter)
    if radar_xy.shape[0] > 0:
        ax.scatter(radar_xy[:, 0], radar_xy[:, 1], s=8)
        step = max(1, radar_xy.shape[0] // 80)
        for i in range(0, radar_xy.shape[0], step):
            x, y = radar_xy[i]
            vx, vy = radar_vxy[i]
            ax.arrow(x, y, vx, vy, head_width=0.7, length_includes_head=True, alpha=0.8)

    # cam gt detections
    if len(cam_dets) > 0:
        cx = [d.x for d in cam_dets]
        cy = [d.y for d in cam_dets]
        ax.scatter(cx, cy, s=28, marker="x")

    # tracks
    for t in tracks:
        ax.scatter([t.x], [t.y], s=60, marker="o")
        ax.text(t.x + 0.8, t.y + 0.8, f"ID{t.track_id}", fontsize=8)
        ax.arrow(t.x, t.y, t.vx, t.vy, head_width=0.9, length_includes_head=True, alpha=0.9)

        # history tail
        if len(t.history) > 2:
            h = np.array(t.history[-20:], dtype=np.float64)
            ax.plot(h[:, 0], h[:, 1], linewidth=1.0, alpha=0.7)

    ax.set_title("BEV (ego frame)")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
