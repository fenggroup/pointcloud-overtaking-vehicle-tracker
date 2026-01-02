"""
main.py

High-level pipeline runner.

Assumes:
- PCD frames in data/pointclouds/frame-******.pcd
- Optionally: outputs/bike_speed_icp.csv already exists (bike speed from ICP)

Pipeline:
0) (optional) bike_speed_estimation.py / bike_speed_icp.py
       → results/bike_speed_icp.csv   (run if missing or if user chooses to recompute)
1) vehicle_tracking.py
       → results/vehicle_tracking_frames_raw.csv
2) passing_events.py
       → results/passing_events_summary.csv
3) smoothing.py
       → results/vehicle_tracking_frames_smoothed.csv
4) speed_per_frame.py
       → results/vehicle_speeds_frames.csv
5) surrogate_safety_metrics.py
       → results/surrogate_safety_metrics.csv
"""

import os

from src.speeds_computation.bike_speed_estimation import main as run_bike_speed_estimation
from src.detection_tracking.vehicle_tracking import run_vehicle_tracking
from src.extracting_passing_events.passing_events import summarize_passing_events
from src.smoothing.smoothing import smooth_tracking_and_bike
from src.speeds_computation.vehicle_speed_per_frame import compute_vehicle_speeds_per_frame
from src.metrics.surrogate_safety_metrics import compute_surrogate_safety_metrics


BIKE_SPEED_CSV = "outputs/bike-speed-per-frame/bike-speed-estimation.csv"


def bike_speed_bool(prompt: str, default: bool = False) -> bool:
    """
    Simple interactive Y/N question.

    default=False → [y/N]
    default=True  → [Y/n]
    """
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        ans = input(f"{prompt} {suffix} ").strip().lower()
        if ans == "" and default is not None:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")


def run_full_pipeline():
    # ── Step 0: Bike speed estimation (optional / conditional) ───────────────
    if not os.path.exists(BIKE_SPEED_CSV):
        print(f"⚠️ Bike speed file '{BIKE_SPEED_CSV}' not found.")
        print("   We need to compute bike speeds first via ICP.")
        run_bike_speed_estimation()
    else:
        print(f"✅ Found existing bike speed file: '{BIKE_SPEED_CSV}'")
        recompute = bike_speed_bool(
            "Do you want to recompute bike speeds (this can take a while)?",
            default=False,
        )
        if recompute:
            run_bike_speed_estimation()
        else:
            print("➤ Skipping bike speed estimation and using existing file.")

    # ── Step 1: Vehicle tracking ─────────────────────────────────────────────
    print("\n==== Step 1: Vehicle tracking & box fitting ====")
    run_vehicle_tracking()

    # ── Step 2: Passing events ───────────────────────────────────────────────
    print("\n==== Step 2: Passing events summary ====")
    summarize_passing_events()

    # ── Step 3: Smoothing (tracking, and any bike-related smoothing if used) ─
    print("\n==== Step 3: Smoothing tracking & bike speed ====")
    smooth_tracking_and_bike()

    # ── Step 4: Per-frame speeds ─────────────────────────────────────────────
    print("\n==== Step 4: Per-frame vehicle speeds (passing only) ====")
    compute_vehicle_speeds_per_frame()

    # ── Step 5: Surrogate safety metrics ─────────────────────────────────────
    print("\n==== Step 5: Surrogate safety metrics (event-level) ====")
    compute_surrogate_safety_metrics()

    print("\n✅ Pipeline complete. Check the 'outputs/' folder.")


if __name__ == "__main__":
    run_full_pipeline()




# """
# main.py

# High-level pipeline runner.

# Assumes:
# - PCD frames in data/pointclouds/frame-******.pcd
# - bike_speed_icp.py has been run to produce results/bike_speed_icp.csv

# Pipeline:
# 1) vehicle_tracking.py        → results/vehicle_tracking_frames_raw.csv
# 2) passing_events.py          → results/passing_events_summary.csv
# 3) smoothing.py               → results/vehicle_tracking_frames_smoothed.csv
#                                  and results/bike_speed_icp_smoothed.csv
# 4) speed_per_frame.py         → results/vehicle_speeds_frames.csv
# 5) surrogate_safety_metrics.py→ results/surrogate_safety_metrics.csv
# """

# from vehicle_tracking import run_vehicle_tracking
# from passing_events import compute_passing_events
# from smoothing import smooth_tracking_and_bike
# from speed_per_frame import compute_vehicle_speeds_per_frame
# from surrogate_safety_metrics import compute_surrogate_safety_metrics


# def run_full_pipeline():
#     print("==== Step 1: Vehicle tracking & box fitting ====")
#     run_vehicle_tracking()

#     print("\n==== Step 2: Passing events summary ====")
#     compute_passing_events()

#     print("\n==== Step 3: Smoothing tracking & bike speed ====")
#     smooth_tracking_and_bike()

#     print("\n==== Step 4: Per-frame vehicle speeds (passing only) ====")
#     compute_vehicle_speeds_per_frame()

#     print("\n==== Step 5: Surrogate safety metrics (event-level) ====")
#     compute_surrogate_safety_metrics()

#     print("\n✅ Pipeline complete. Check the 'results/' folder.")


# if __name__ == "__main__":
#     run_full_pipeline()

