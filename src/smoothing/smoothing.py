
"""
smoothing.py

- Interpolate + smooth per-frame vehicle tracking results

"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

TRACKING_IN = "outputs/detection-and-tracking/vehicle-tracking-frames-raw.csv"
TRACKING_OUT = "outputs/detection-and-tracking/vehicle-tracking-frames-smoothed.csv"

SG_WIN = 11   # Savitzky–Golay window (frames, odd)
SG_POLY = 3   # polynomial order
DECIMALS = 2  # rounding precision


def smooth_tracking_and_bike(tracking_in: str = TRACKING_IN,
                             bike_in: str = None,
                             tracking_out: str = TRACKING_OUT,
                             bike_out: str = None):
    """
    Interpolate + smooth tracking series per vehicle.

    bike_in / bike_out arguments are kept for backward compatibility but
    are ignored, since bike speed smoothing is now done in bike_speed_icp.py.
    """
    # ────────── TRACKING ──────────
    df_track = pd.read_csv(tracking_in).sort_values(["vehicle_id", "frame"])
    df_track = df_track.drop_duplicates(subset=["vehicle_id", "frame"])

    exclude_cols = ["vehicle_id", "frame"]
    smooth_cols = [
        c for c in df_track.columns
        if c not in exclude_cols and df_track[c].dtype.kind in "fc"
    ]

    all_rows = []

    print("========== Interpolated frames per vehicle ==========")

    for vid, g in df_track.groupby("vehicle_id"):
        g = g.sort_values("frame").copy()
        frames_full = np.arange(g["frame"].min(), g["frame"].max() + 1)
        g_full = g.set_index("frame").reindex(frames_full).reset_index()
        g_full.rename(columns={"index": "frame"}, inplace=True)
        g_full["vehicle_id"] = vid

        missing = g_full[g_full[smooth_cols].isna().any(axis=1)]["frame"].tolist()
        if missing:
            print(f"Vehicle {vid:3d}: added frames {missing}")
        else:
            print(f"Vehicle {vid:3d}: no gaps")

        for col in smooth_cols:
            g_full[col] = g_full[col].interpolate(method="linear")
            if len(g_full) >= SG_WIN:
                g_full[col] = savgol_filter(
                    g_full[col], window_length=SG_WIN, polyorder=SG_POLY
                )

        all_rows.append(g_full[["vehicle_id", "frame"] + smooth_cols])

    df_track_smoothed = pd.concat(all_rows, ignore_index=True)

    # Merge back with original to retain any non-float columns
    df_final = (
        df_track.drop(columns=smooth_cols)
                .merge(df_track_smoothed, on=["vehicle_id", "frame"], how="outer")
                .sort_values(["vehicle_id", "frame"])
                .reset_index(drop=True)
    )

    float_cols = df_final.select_dtypes(include=["float"]).columns
    df_final[float_cols] = df_final[float_cols].round(DECIMALS)

    os.makedirs(os.path.dirname(tracking_out), exist_ok=True)
    df_final.to_csv(tracking_out, index=False)
    print(f"\n✅ Smoothed tracking file saved to '{tracking_out}'")


if __name__ == "__main__":
    smooth_tracking_and_bike()
