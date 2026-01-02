
"""
speed_per_frame.py

Compute frame-level relative and absolute vehicle speeds for passing vehicles,
based on smoothed tracking and smoothed bike speed from bike_speed_icp.csv.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

TRACKING_FILE   = "outputs/detection-and-tracking/vehicle-tracking-frames-smoothed.csv"
BIKE_SPEED_FILE = "outputs/bike-speed-per-frame/bike-speed-estimation.csv"        
EVENTS_FILE     = "outputs/passing-events/passing-events-summary.csv"
OUTPUT_CSV      = "outputs/vehicles-speed/vehicle-speeds-frames.csv"

FPS        = 10.0
MPS_TO_MPH = 2.23694
WIN3       = 3              # window in frames for finite difference
SG_WIN     = 11
SG_POLY    = 2


def _compute_speed(group: pd.DataFrame) -> pd.DataFrame:
    dt = WIN3 / FPS

    x = group["box_center_x"].astype(float)
    bike = group["bike_speed_mph"].astype(float)   # this will be the SMOOTHED bike speed

    dx = x - x.shift(WIN3)
    rel = (dx / dt) * MPS_TO_MPH          # relative speed (mph)
    abs_v = -rel + bike                   # absolute vehicle speed in mph

    # --- Smoothing ---
    if rel.notna().sum() >= SG_WIN:
        rel_filled = rel.ffill().bfill()
        rel_smooth = pd.Series(
            savgol_filter(rel_filled, SG_WIN, SG_POLY, mode="interp"),
            index=rel.index,
        )
    else:
        rel_smooth = rel

    if abs_v.notna().sum() >= SG_WIN:
        abs_filled = abs_v.ffill().bfill()
        abs_smooth = pd.Series(
            savgol_filter(abs_filled, SG_WIN, SG_POLY, mode="interp"),
            index=abs_v.index,
        )
    else:
        abs_smooth = abs_v

    return pd.DataFrame(
        {
            "delta_x_m": dx,
            "delta_t_s": dt,
            "relative_speed_mph": rel,
            "vehicle_speed_mph": abs_v,
            "relative_speed_smoothed_mph": rel_smooth,
            "vehicle_speed_smoothed_mph": abs_smooth,
        },
        index=group.index,
    )



def compute_vehicle_speeds_per_frame(
    tracking_file: str   = TRACKING_FILE,
    bike_speed_file: str = BIKE_SPEED_FILE,
    events_file: str     = EVENTS_FILE,
    out_csv: str         = OUTPUT_CSV,
):
    # ── 1) Events → passing vehicle IDs ───────────────────────────────────────
    events_df = pd.read_csv(events_file)
    passing_ids = events_df.loc[events_df["status"] == "passing", "vehicle_id"].unique()

    if len(passing_ids) == 0:
        print("⚠️ No vehicles with status == 'passing' in events file. "
              "Output will be empty, but code will still run.")

    # ── 2) Load tracking ─────────────────────────────────────────────────────
    tracking_df = (
        pd.read_csv(tracking_file)
        .sort_values(["vehicle_id", "frame"])
        .reset_index(drop=True)
    )

    # ── 3) Load bike speed and map to frame ──────────────────────────────────
    # bike_speed_icp.csv now contains:
    #   start_frame, end_frame, bike_displacement_m, bike_speed_mph, bike_speed_smoothed_mph
    bdf_raw = pd.read_csv(bike_speed_file)

    # If smoothed column exists, use it as our bike_speed_mph
    if "bike_speed_smoothed_mph" in bdf_raw.columns:
        bdf_raw["bike_speed_mph"] = bdf_raw["bike_speed_smoothed_mph"]

    bdf = (
        bdf_raw
        .rename(columns={"end_frame": "frame"})  # associate speed to 'frame'
        [["frame", "bike_speed_mph"]]
        .drop_duplicates("frame")
    )

    df = tracking_df.merge(bdf, on="frame", how="left")

    # ── 4) Filter to passing vehicles only (may be empty for small samples) ──
    df_pass = df[df["vehicle_id"].isin(passing_ids)].copy()

    if df_pass.empty:
        print("⚠️ No rows corresponding to passing vehicles in this subset. "
              "Writing empty result file.")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        # write a header-only CSV
        empty_cols = [
            "frame",
            "vehicle_id",
            "is_passing_frame",
            "box_center_x",
            "bike_speed_mph",
            "delta_x_m",
            "delta_t_s",
            "relative_speed_mph",
            "vehicle_speed_mph",
            "relative_speed_smoothed_mph",
            "vehicle_speed_smoothed_mph",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(out_csv, index=False)
        print(f"✅ Saved (empty) per-frame vehicle speed results to '{out_csv}'")
        return

    # ── 5) Compute speeds per vehicle (groupby apply) ────────────────────────
    speed_df = (
        df_pass.groupby("vehicle_id", group_keys=False)[
            ["frame", "box_center_x", "bike_speed_mph"]
        ]
        .apply(_compute_speed)
    )

    # Concatenate side-by-side using index alignment
    result_df = pd.concat([df_pass, speed_df], axis=1)

    # ── 6) Select & round columns ────────────────────────────────────────────
    cols = [
        "frame",
        "vehicle_id",
        "is_passing_frame",
        "box_center_x",
        "bike_speed_mph",
        "delta_x_m",
        "delta_t_s",
        "relative_speed_mph",
        "vehicle_speed_mph",
        "relative_speed_smoothed_mph",
        "vehicle_speed_smoothed_mph",
    ]
    cols = [c for c in cols if c in result_df.columns]
    result_df = result_df[cols]

    round_cols = [
        c
        for c in result_df.columns
        if (c in ["delta_x_m", "delta_t_s"]) or ("speed" in c)
    ]
    if round_cols:
        result_df[round_cols] = result_df[round_cols].round(2)

    # ── 7) Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    result_df.to_csv(out_csv, index=False)
    print(f"✅ Saved per-frame vehicle speed results to '{out_csv}'")


if __name__ == "__main__":
    compute_vehicle_speeds_per_frame()
