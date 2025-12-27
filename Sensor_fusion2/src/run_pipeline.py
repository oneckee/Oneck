# src/run_pipeline.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# ---- Make imports work when running: python src/run_pipeline.py
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse

from nuscenes.nuscenes import NuScenes

from fusion.object_pipeline import PipelineConfig, ObjectPipeline
from visualize_bev import render_bev_frame


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--version", type=str, default="v1.0-mini")
    p.add_argument(
        "--dataroot",
        type=str,
        default=str(PROJECT_ROOT / "data" / "nuscenes"),
        help="Path to data/nuscenes (NOT to v1.0-mini directly)",
    )
    p.add_argument("--scene_idx", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=40)
    p.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "output" / "bev_frames"))
    p.add_argument("--dpi", type=int, default=130)
    return p.parse_args()


def main():
    args = parse_args()

    dataroot = Path(args.dataroot).expanduser().resolve()
    version_dir = dataroot / args.version

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_ROOT   :", dataroot)
    print("VERSION_DIR :", version_dir)

    # ---- HARD FAIL (빈 프레임 저장 방지)
    assert dataroot.exists(), f"dataroot not found: {dataroot}"
    assert version_dir.exists(), f"Database version not found: {version_dir}\n" \
                                 f"Expected: <dataroot>/{args.version} (e.g. data/nuscenes/v1.0-mini)"

    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=False)

    # ---- pick scene
    assert args.scene_idx < len(nusc.scene), f"scene_idx out of range: {args.scene_idx}"
    scene = nusc.scene[args.scene_idx]
    first_sample_token = scene["first_sample_token"]
    print(f"🎬 scene={scene['name']} ({args.scene_idx}) first_sample_token={first_sample_token}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("📦 output_dir:", out_dir)

    cfg = PipelineConfig()
    pipe = ObjectPipeline(cfg)

    # ---- iterate samples
    token = first_sample_token
    frame_i = 0
    while token:
        sample = nusc.get("sample", token)

        frame = pipe.process(nusc, sample)

        # --- sanity print (여기 숫자가 0이면 로더 문제)
        print(
            f"[frame {frame_i:03d}] "
            f"lidar_pts={frame['lidar_xy'].shape[0]} "
            f"radar_pts={frame['radar_xy'].shape[0]} "
            f"radar_obj={len(frame.get('detections', []))} "
            f"tracks={len(frame.get('tracks', []))}"

            # f"cam_dets={len(frame['cam_dets'])} "
            # f"tracks={len(frame['tracks'])}"
        )

    


        out_png = out_dir / f"frame_{frame_i:04d}.png"
        render_bev_frame(frame, save_path=str(out_png), dpi=args.dpi)

        frame_i += 1
        if frame_i >= args.max_frames:
            break

        token = sample["next"]

    print(f"✅ done. saved {frame_i} frames to: {out_dir}")


if __name__ == "__main__":
    main()
