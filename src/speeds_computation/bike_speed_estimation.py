"""
bike_speed_estimation.py

Estimate bike speed via frame-to-frame registration of LiDAR point clouds
using FPFH + RANSAC + ICP.

This script is intentionally standalone so it can be run independently
on a dataset to produce bike_speed_icp.csv.
"""

import csv
import os
import concurrent.futures
import threading
import sys

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.signal import savgol_filter

from ..pcd_io import load_o3d_from_pcd

csv_lock = threading.Lock()

# Default config 
PCD_DIR = "data/pointclouds"
OUT_CSV = "outputs/bike-speed-per-frame/bike-speed-estimation.csv"
VOXEL_SIZE = 0.2
FPS = 10.0
FRAME_FORMAT = "frame-{0:06d}.pcd"

# Optional skip ranges (same idea as in tracking; default none)
SKIP_FRAME_RANGES = []

# Outlier + smoothing config
MAX_DISP_M = 5.0          # max reasonable bike displacement between frames (m)
MAX_SPEED_MPH = 40.0      # max reasonable bike speed (mph)
SG_WIN = 11               # Savitzky–Golay window (frames, must be odd)
SG_POLY = 3               # Savitzky–Golay polynomial order


def load_and_crop_point_cloud(file_path,
                              x_range=None,
                              y_range=None,
                              z_range=None):
    """
    Load a PCD file as Open3D point cloud and crop it to the given ranges
    (if provided).
    """
    pcd = load_o3d_from_pcd(file_path)
    if x_range or y_range or z_range:
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[
                x_range[0] if x_range else -np.inf,
                y_range[0] if y_range else -np.inf,
                z_range[0] if z_range else -np.inf,
            ],
            max_bound=[
                x_range[1] if x_range else np.inf,
                y_range[1] if y_range else np.inf,
                z_range[1] if z_range else np.inf,
            ],
        )
        pcd = pcd.crop(bbox)
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    """
    Downsample the point cloud and estimate normals.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30
        )
    )
    return pcd_down


def compute_fpfh_features(pcd, voxel_size):
    """
    Compute Fast Point Feature Histogram (FPFH) features.
    """
    pcd_down = preprocess_point_cloud(pcd, voxel_size)
    radius_feature = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100
        )
    )
    return pcd_down, fpfh


def global_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    """
    Perform RANSAC-based global registration.
    """
    distance_threshold = voxel_size * 2.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target,
        source_fpfh, target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
    )
    return result


def refine_registration(source, target, init_transformation, voxel_size):
    """
    Refine registration using point-to-plane ICP.
    """
    distance_threshold = voxel_size * 0.4
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        distance_threshold,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result_icp


def is_in_skip_ranges(i, skip_ranges):
    """
    Returns True if frame i falls within any of the skip ranges.
    """
    for start, end in skip_ranges:
        if start <= i <= end:
            return True
    return False


def register_pair(source, target, voxel_size=VOXEL_SIZE):
    """
    Full registration: global (RANSAC) + ICP refinement.
    """
    source_down, source_fpfh = compute_fpfh_features(source, voxel_size)
    target_down, target_fpfh = compute_fpfh_features(target, voxel_size)

    result_ransac = global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    result_icp = refine_registration(
        source_down, target_down, result_ransac.transformation, voxel_size
    )

    return result_icp.transformation


def process_pair(i, pcd_dir, voxel_size, fps, frame_format,
                 out_csv,
                 x_range=None, y_range=None, z_range=None):
    """
    Process a single frame pair i -> i+1.
    Writes raw displacement & speed to CSV (outlier handling is done later).
    """
    src = os.path.join(pcd_dir, frame_format.format(i))
    tgt = os.path.join(pcd_dir, frame_format.format(i + 1))
    if not (os.path.exists(src) and os.path.exists(tgt)):
        # Missing file, just skip
        return None

    try:
        source = load_and_crop_point_cloud(src, x_range, y_range, z_range)
        target = load_and_crop_point_cloud(tgt, x_range, y_range, z_range)
        T = register_pair(source, target, voxel_size)

        disp = float(np.linalg.norm(T[:3, 3]))
        speed_mps = disp * fps
        speed_mph = speed_mps * 2.237

        disp_rounded = round(disp, 2)
        speed_rounded = round(speed_mph, 1)

        # We still write raw values; real outlier removal happens in postprocess.
        with csv_lock:
            with open(out_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([i, i + 1, disp_rounded, speed_rounded])

    except Exception as e:
        
        print(f"\nError processing frame {i}: {e}", file=sys.stderr)

    return "OK"



def _print_progress(done, total):
    if total == 0:
        return
    frac = done / total
    bar_len = 30
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    msg = f"Processing frame pairs: [{bar}] {done}/{total} ({frac*100:5.1f}%)"


    if sys.stdout.isatty():
        print("\r" + msg, end="", flush=True)
    else:
        # Fallback: print one line per update (no \r)
        print(msg, flush=True)




def compute_pairwise_speeds_parallel(
    pcd_dir: str,
    start_frame: int,
    end_frame: int,
    out_csv: str,
    voxel_size: float = VOXEL_SIZE,
    fps: float = FPS,
    frame_format: str = FRAME_FORMAT,
    x_range=None,
    y_range=None,
    z_range=None,
    skip_ranges=None,
):
    if skip_ranges is None:
        skip_ranges = []

    valid_indices = [
        i for i in range(start_frame, end_frame)
        if not is_in_skip_ranges(i, skip_ranges)
    ]

    total = len(valid_indices)
    if total == 0:
        print("No frame pairs to process.")
        return

    print(f"✅ Will process {total} frame pairs...")
    
    processed = 0
    _print_progress(processed, total)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                process_pair,
                i, pcd_dir, voxel_size, fps, frame_format, out_csv,
                x_range, y_range, z_range,
            )
            for i in valid_indices
        ]
    
        for _ in concurrent.futures.as_completed(futures):
            processed += 1
            _print_progress(processed, total)
    
    print("\n✅ Done processing frame pairs.")




def postprocess_bike_speed_csv(out_csv: str):
    """
    - Load raw bike_speed_icp.csv
    - Flag outliers (huge displacement/speed) as NaN
    - Interpolate speeds (and displacement) over NaNs
    - Apply Savitzky–Golay smoothing
    - Save back with an extra column: bike_speed_smoothed_mph
    """
    if not os.path.exists(out_csv):
        print(f"⚠️ Cannot postprocess; file not found: {out_csv}")
        return

    df = pd.read_csv(out_csv)

    if df.empty:
        print("⚠️ bike_speed_icp.csv is empty; nothing to postprocess.")
        return

    # Ensure ordering by start_frame
    df = df.sort_values("start_frame").reset_index(drop=True)

    # Rename columns if needed to consistent names
    if "displacement (m)" in df.columns:
        df.rename(columns={
            "displacement (m)": "bike_displacement_m",
            "instant speed (mph)": "bike_speed_mph"
        }, inplace=True)

    # Ensure expected columns exist
    expected_cols = {"start_frame", "end_frame", "bike_displacement_m", "bike_speed_mph"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected column(s) in {out_csv}: {missing}")

    # ---- Outlier flagging ----
    # Any absurd displacement OR speed is flagged as NaN and later interpolated.
    mask_outlier = (
        (df["bike_displacement_m"].abs() > MAX_DISP_M) |
        (df["bike_speed_mph"].abs() > MAX_SPEED_MPH)
    )

    if mask_outlier.any():
        outlier_frames = df.loc[mask_outlier, "start_frame"].tolist()
        print(f"⚠️ Flagging outliers at frames: {outlier_frames}")
        df.loc[mask_outlier, ["bike_displacement_m", "bike_speed_mph"]] = np.nan

    # ---- Interpolate over NaNs ----
    df["bike_displacement_m"] = df["bike_displacement_m"].interpolate(
        method="linear", limit_direction="both"
    )
    df["bike_speed_mph"] = df["bike_speed_mph"].interpolate(
        method="linear", limit_direction="both"
    )

    # ---- Smoothing (Savitzky–Golay) ----
    if len(df) >= SG_WIN:
        df["bike_speed_smoothed_mph"] = savgol_filter(
            df["bike_speed_mph"].values,
            window_length=SG_WIN,
            polyorder=SG_POLY,
            mode="interp",
        )
    else:
        df["bike_speed_smoothed_mph"] = df["bike_speed_mph"]

    # ---- Rounding ----
    df["bike_displacement_m"] = df["bike_displacement_m"].round(2)
    df["bike_speed_mph"] = df["bike_speed_mph"].round(2)
    df["bike_speed_smoothed_mph"] = df["bike_speed_smoothed_mph"].round(2)

    # Save back
    df.to_csv(out_csv, index=False)
    print(f"✅ Post-processed bike speeds (outliers removed + smoothed) → '{out_csv}'")


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # Write header (raw ICP results)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_frame", "end_frame", "bike_displacement_m", "bike_speed_mph"])

    # Example: user should set start_frame / end_frame pre-known interval
    start_frame = 1950
    end_frame = 2100

    # 1) Run pairwise ICP speeds (raw)
    compute_pairwise_speeds_parallel(
        pcd_dir=PCD_DIR,
        start_frame=start_frame,
        end_frame=end_frame,
        out_csv=OUT_CSV,
        voxel_size=VOXEL_SIZE,
        fps=FPS,
        frame_format=FRAME_FORMAT,
        skip_ranges=SKIP_FRAME_RANGES,
    )

    # 2) Post-process: outlier removal + interpolation + smoothing
    postprocess_bike_speed_csv(OUT_CSV)


if __name__ == "__main__":
    main()

