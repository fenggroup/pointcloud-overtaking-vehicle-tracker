"""
passing_events.py

Compute per-vehicle passing events from per-frame tracking data.
"""

import os
import pandas as pd

TRACKING_FILE = "outputs/detection-and-tracking/vehicle-tracking-frames-raw.csv"
OUT_CSV = "outputs/passing-events/passing-events-summary.csv"
FPS = 10.0

PASS_FRONT_X_THRESH = 0.27
PASS_REAR_X_THRESH = -1.5


def summarize_passing_events(tracking_file: str = TRACKING_FILE,
                           out_csv: str = OUT_CSV,
                           fps: float = FPS):
    df = (
        pd.read_csv(tracking_file)
        .sort_values(["vehicle_id", "frame"])
        .reset_index(drop=True)
    )

    events = []

    for vehicle_id, g in df.groupby("vehicle_id"):
        g = g.sort_values("frame").reset_index(drop=True)

        print(f"\nProcessing vehicle_id {vehicle_id} with {len(g)} frames")

        track_start_frame = int(g.iloc[0]["frame"])
        track_end_frame = int(g.iloc[-1]["frame"])

        # Passing window (frames marked as passing)
        pass_window = g[g["is_passing_frame"] == 1]

        if not pass_window.empty:
            pass_start_frame = int(pass_window.iloc[0]["frame"])
            pass_end_frame = int(pass_window.iloc[-1]["frame"])
        else:
            pass_start_frame = None
            pass_end_frame = None

        print(f"→ Passing window: {pass_start_frame} → {pass_end_frame}")

        # Status flags 
        front_entered = (
            (g["front_x"] > PASS_FRONT_X_THRESH).astype(int).diff() == -1
        ).any()

        rear_exited = (
            (g["rear_x"] > PASS_REAR_X_THRESH).astype(int).diff() == -1
        ).any()

        if front_entered and rear_exited:
            status = "passing"
        elif front_entered or rear_exited:
            status = "partial_pass"
        else:
            status = "no_passing"

        # Minimum distance over the entire track
        min_val = g["min_distance_m"].min()
        idx_min = g["min_distance_m"].idxmin()

        try:
            min_row = g.loc[idx_min]
            passing_frame = int(min_row["frame"])
            print(f"✅ Minimum distance = {min_val:.3f} m at frame {passing_frame}")
        except Exception as e:
            print(f"❌ Failed to extract min row for vehicle {vehicle_id}: {e}")
            passing_frame = None
            min_val = None

        # Passing duration in seconds
        if pass_start_frame is not None and pass_end_frame is not None:
            duration_frames = pass_end_frame - pass_start_frame + 1
            passing_duration_s = round(duration_frames / fps, 2)
        else:
            passing_duration_s = None

        events.append({
            "vehicle_id": vehicle_id,
            "track_start_frame": track_start_frame,
            "track_end_frame": track_end_frame,
            "pass_start_frame": pass_start_frame,
            "pass_end_frame": pass_end_frame,
            "passing_frame": passing_frame,
            "passing_duration_s": passing_duration_s,
            "passing_distance_m": round(min_val, 3) if min_val is not None else None,
            "status": status,
        })

        # --- SAVE ---
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    
        df_events = pd.DataFrame(events)
    
        # Force integer frame numbers but allow NaNs
        int_cols = [
            "vehicle_id",
            "track_start_frame",
            "track_end_frame",
            "pass_start_frame",
            "pass_end_frame",
            "passing_frame",
        ]
    
        for col in int_cols:
            df_events[col] = df_events[col].astype("Int64")
    
        df_events.to_csv(out_csv, index=False)
    
        print(f"\n✅ Saved {len(events)} event records to '{out_csv}'")


if __name__ == "__main__":
    compute_passing_events()
