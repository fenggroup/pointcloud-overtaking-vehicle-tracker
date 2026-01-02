"""
box_fitting.py

High-level postprocessing:
- Fit hugged axis-aligned bounding boxes per vehicle
- Enforce consistent per-vehicle box dimensions
- Detect passing frames using front_x / rear_x thresholds
- Filter out very short tracks
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def fit_hugged_box(pts: np.ndarray,
                   min_dims: np.ndarray):
    """
    Fit an axis-aligned bounding box and 'hug' either the front or rear face
    depending on the vehicle's longitudinal position relative to the LiDAR.

    Returns
    -------
    new_center : np.ndarray
        Box center [x, y, z].
    ext : np.ndarray
        Box extents [L, W, H] (meters).
    face : str
        "front" or "rear".
    face_x : float
        X coordinate of the hugged face.
    """
    ctr = pts.mean(0)
    loc = pts - ctr
    mn, mx = loc.min(0), loc.max(0)
    ext = np.maximum(mx - mn, min_dims)

    x_min_w, x_max_w = (ctr + mn)[0], (ctr + mx)[0]

    if ctr[0] > 0:  # car is behind → hug front
        new_cx = x_min_w + ext[0] / 2
        face = "front"
        face_x = x_min_w
    else:           # car is ahead → hug rear
        new_cx = x_max_w - ext[0] / 2
        face = "rear"
        face_x = x_max_w

    new_center = np.array([
        new_cx,
        ctr[1] + (mn[1] + mx[1]) / 2,
        ctr[2] + (mn[2] + mx[2]) / 2
    ])
    return new_center, ext, face, face_x


def postprocess_tracking_records(
    tracking_records: List[dict],
    min_dims: np.ndarray,
    lane_width: float,
    pass_front_x_thresh: float,
    pass_rear_x_thresh: float,
    min_track_frames: int,
) -> pd.DataFrame:
    """
    From raw tracking records with cluster points:

    1) Fit hugged boxes and keep only plausible lane-width vehicles
    2) Enforce consistent box dimensions per vehicle
    3) Detect passing start/end and mark is_passing_frame
    4) Filter out very short tracks
    5) Renumber vehicle_ids sequentially from 0
    """
    if not tracking_records:
        return pd.DataFrame()

    interim: List[dict] = []
    max_dims: Dict[int, np.ndarray] = {}

    # 1) First pass: fit a box each time and record max extents per vehicle_id
    for rec in tracking_records:
        pts = rec["points"]
        if pts.size == 0:
            continue

        center, ext, face, face_x = fit_hugged_box(pts, min_dims=min_dims)
        if ext[1] > lane_width:
            # Skip boxes that are wider than the lane
            continue

        vid = rec["vehicle_id"]
        max_dims[vid] = np.maximum(max_dims.get(vid, min_dims), ext)

        interim.append({
            "frame": rec["frame"],
            "vehicle_id": vid,
            "centroid": rec["centroid"],
            "min_distance_m": rec["min_distance_m"],
            "face": face,
            "face_x": face_x,
            "cy": center[1],
            "cz": center[2],
        })

    if not interim:
        return pd.DataFrame()

    grouped: Dict[int, List[dict]] = {}
    out_rows: List[dict] = []

    # 2) Second pass: apply consistent box size and anchor face per vehicle
    for rec in interim:
        vid = rec["vehicle_id"]
        ext = max_dims[vid]
        face = rec["face"]
        face_x = rec["face_x"]

        if face == "front":
            cx = face_x + ext[0] / 2.0
        else:  # "rear"
            cx = face_x - ext[0] / 2.0

        center = np.array([cx, rec["cy"], rec["cz"]])
        front_x = cx - ext[0] / 2.0
        rear_x = cx + ext[0] / 2.0

        row = {
            "frame": int(rec["frame"]),
            "vehicle_id": int(vid),
            "centroid_x": round(float(rec["centroid"][0]), 3),
            "centroid_y": round(float(rec["centroid"][1]), 3),
            "centroid_z": round(float(rec["centroid"][2]), 3),
            "min_distance_m": float(rec["min_distance_m"]),
            "box_center_x": round(float(center[0]), 3),
            "box_center_y": round(float(center[1]), 3),
            "box_center_z": round(float(center[2]), 3),
            "box_length_m": round(float(ext[0]), 3),
            "box_width_m": round(float(ext[1]), 3),
            "box_height_m": round(float(ext[2]), 3),
            "front_x": round(float(front_x), 3),
            "rear_x": round(float(rear_x), 3),
        }

        grouped.setdefault(vid, []).append(row)

    final_rows: List[dict] = []

    # 3) Detect passing start/end per vehicle and mark is_passing_frame
    for vid, car_rows in grouped.items():
        car_rows.sort(key=lambda r: r["frame"])

        pass_start = None
        pass_end = None

        # First frame where front_x exceeds threshold → vehicle in front
        for r in car_rows:
            if r["front_x"] > pass_front_x_thresh:
                pass_start = r["frame"]

        # First frame where rear_x drops below threshold → fully passed
        for r in car_rows:
            if r["rear_x"] < pass_rear_x_thresh:
                pass_end = r["frame"]
                break

        for r in car_rows:
            r["is_passing_frame"] = int(
                pass_start is not None and
                pass_end is not None and
                pass_start <= r["frame"] <= pass_end
            )
            final_rows.append(r)

    df_out = pd.DataFrame(final_rows).sort_values(["vehicle_id", "frame"])

    if df_out.empty:
        return df_out

    # 4) Filter out very short tracks
    track_lengths = (
        df_out.groupby("vehicle_id")["frame"]
        .nunique()
        .rename("n_frames")
    )
    keep_ids = track_lengths[track_lengths >= min_track_frames].index

    if len(keep_ids) == 0:
        return pd.DataFrame()

    df_out = df_out[df_out["vehicle_id"].isin(keep_ids)].copy()

    # 5) Renumber vehicle IDs to be consecutive starting from 0
    unique_ids = sorted(df_out["vehicle_id"].unique())
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    df_out["vehicle_id"] = df_out["vehicle_id"].map(id_map)

    return df_out
