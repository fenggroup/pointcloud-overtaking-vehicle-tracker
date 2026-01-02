"""
surrogate_safety_metrics.py

Aggregate event-level surrogate safety metrics from:
- passing-events-summary.csv
- vehicle-speeds-frames.csv
"""

import os
import pandas as pd

EVENTS_FILE = "outputs/passing-events/passing-events-summary.csv"
SPEEDS_FILE = "outputs/vehicles-speed/vehicle-speeds-frames.csv"
OUT_CSV = "outputs/metrics/surrogate-safety-metrics.csv"


def compute_surrogate_safety_metrics(events_file: str = EVENTS_FILE,
                                     speeds_file: str = SPEEDS_FILE,
                                     out_csv: str = OUT_CSV):

    events_df = pd.read_csv(events_file)
    speeds_df = pd.read_csv(speeds_file)

    # Only consider full passing events
    passing_events = events_df[events_df["status"] == "passing"].copy()

    rows = []

    for _, ev in passing_events.iterrows():
        vid = ev["vehicle_id"]
        start_f = int(ev["pass_start_frame"])
        end_f   = int(ev["pass_end_frame"])
        min_f   = int(ev["passing_frame"])

        # Slice speeds for this event
        seg = speeds_df[
            (speeds_df["vehicle_id"] == vid) &
            (speeds_df["frame"] >= start_f) &
            (speeds_df["frame"] <= end_f)
        ]

        if seg.empty:
            bike_mean = rel_mean = veh_mean = None
        else:
            bike_mean = seg["bike_speed_mph"].mean()
            rel_mean  = seg["relative_speed_smoothed_mph"].mean()
            veh_mean  = -rel_mean + bike_mean

        seg_min = speeds_df[
            (speeds_df["vehicle_id"] == vid) &
            (speeds_df["frame"] == min_f)
        ]

        if not seg_min.empty:
            bike_at_min = seg_min["bike_speed_mph"].iloc[0]
            rel_at_min  = seg_min["relative_speed_smoothed_mph"].iloc[0]
            veh_at_min  = -rel_at_min + bike_at_min
        else:
            bike_at_min = rel_at_min = veh_at_min = None


        rows.append({
            "vehicle_id": vid,
            "pass_start_frame": int(start_f),
            "pass_end_frame": int(end_f),
            "passing_frame": min_f,
            "passing_duration_s": ev["passing_duration_s"],
            "passing_distance_m": ev["passing_distance_m"],
            "bike_speed_mean_mph": None if bike_mean is None else round(bike_mean, 2),
            "bike_speed_at_min_dist_mph": None if bike_at_min is None else round(bike_at_min, 2),
            "relative_speed_mean_mph": None if rel_mean is None else round(rel_mean, 2),
            "relative_speed_at_min_dist_mph": None if rel_at_min is None else round(rel_at_min, 2),
            "vehicle_speed_mean_mph": None if veh_mean is None else round(veh_mean, 2),
            "vehicle_speed_at_min_dist_mph": None if veh_at_min is None else round(veh_at_min, 2),

        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Saved surrogate safety metrics to '{out_csv}'")


if __name__ == "__main__":
    compute_surrogate_safety_metrics()
