"""
vehicle_tracking.py

High-level pipeline orchestrator:
- Runs detection + tracking per frame
- Fits bounding boxes
- Detects passing frames
- Saves cleaned vehicle tracking CSV
- Prints final summary (# vehicles detected)
"""

import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .tracking import (
    reset_global_ids,
    process_frame,
    is_frame_skipped,
)
from .box_fitting import postprocess_tracking_records


# ─── CONFIG ───────────────────────────────────────────────

PCD_DIR = "data/pointclouds"
OUTPUT_CSV = "outputs/detection-and-tracking/vehicle-tracking-frames-raw.csv"

MIN_DIMS = np.array([4.5, 2.0, 1.2])   # [L, W, H]
LANE_W = 3.6
PASS_FRONT_X_THRESH = 0.27
PASS_REAR_X_THRESH  = -1.5

SKIP_FRAME_RANGES: List[Tuple[int, int]] = []

MIN_TRACK_FRAMES = 5


# ─── PROGRESS BAR ─────────────────────────────────────────

def _print_progress(done: int, total: int) -> None:
    """Single-line progress bar."""
    if total == 0:
        return
    frac = done / total
    bar_len = 30
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    msg = f"Processing frames: [{bar}] {done}/{total} ({frac*100:5.1f}%)"

    if sys.stdout.isatty():
        print("\r" + msg, end="", flush=True)
    else:
        print(msg, flush=True)


# ─── MAIN RUNNER ──────────────────────────────────────────

def run_vehicle_tracking(pcd_dir: str = PCD_DIR,
                         output_csv: str = OUTPUT_CSV):
    """Vehicle tracking pipeline with progress bar & final summary only."""

    reset_global_ids()

    # Gather available frame numbers
    files = [f for f in os.listdir(pcd_dir) if f.endswith(".pcd")]
    frame_numbers = []

    for filename in files:
        m = re.search(r"frame-(\d{6})\.pcd", filename)
        if m:
            frame = int(m.group(1))
            if not is_frame_skipped(frame, SKIP_FRAME_RANGES):
                frame_numbers.append(frame)

    frame_numbers.sort()

    if not frame_numbers:
        print("No frames found in dataset.")
        return

    print(f"[Vehicle Tracking] {len(frame_numbers)} frames detected. Starting processing...")

    # Tracking loop
    tracking_records: List[dict] = []
    previous_cars: Dict[int, np.ndarray] = {}
    previous_event_frames: Dict[int, Dict[int, np.ndarray]] = {}

    total_frames = len(frame_numbers)
    processed = 0
    _print_progress(processed, total_frames)

    for frame in frame_numbers:
        frame_recs, previous_cars, previous_event_frames = process_frame(
            frame_number=frame,
            pcd_dir=pcd_dir,
            skip_ranges=SKIP_FRAME_RANGES,
            previous_cars=previous_cars,
            previous_event_frames=previous_event_frames,
        )
        tracking_records.extend(frame_recs)

        processed += 1
        _print_progress(processed, total_frames)

    # Move to next line after progress bar
    if sys.stdout.isatty():
        print()

    if not tracking_records:
        print("No vehicle tracks found. Nothing to save.")
        return

    # Postprocessing: box fitting, passing logic, filtering
    df_out = postprocess_tracking_records(
        tracking_records=tracking_records,
        min_dims=MIN_DIMS,
        lane_width=LANE_W,
        pass_front_x_thresh=PASS_FRONT_X_THRESH,
        pass_rear_x_thresh=PASS_REAR_X_THRESH,
        min_track_frames=MIN_TRACK_FRAMES,
    )

    if df_out.empty:
        print("No valid tracks after postprocessing. Nothing to save.")
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    # Final clean summary
    n_frames = len(frame_numbers)
    n_vehicles = df_out["vehicle_id"].nunique()

    print(f"✅ Vehicle tracking completed.")
    print(f"   Frames processed: {n_frames}")
    print(f"   Vehicles detected: {n_vehicles}")
    print(f"   Output saved to: {output_csv}")

