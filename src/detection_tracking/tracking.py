"""
tracking.py

Mid-level tracking logic:
- Frame skipping logic
- Global ID assignment and nearest-neighbor matching
- Motion filter (longitudinal velocity)
- Per-frame tracking record generation
"""

from typing import Dict, List, Tuple

import numpy as np

from .detection import load_point_cloud_xyz, filter_points, segment_objects

# ─── CONFIG ────────────────────────────────────────────────────────────────

FPS = 10.0               # LiDAR frame rate (Hz)
MIN_SPEED_MPS = -2.0     # minimum avg longitudinal speed (m/s) (negative = approaching)

# Whether to print debug info (keep False for clean pipeline runs)
VERBOSE = False

# Global ID counter
_next_global_car_id: int = 0


def reset_global_ids() -> None:
    """Reset the global vehicle ID counter at the start of each run."""
    global _next_global_car_id
    _next_global_car_id = 0


def match_global_id(centroid: np.ndarray,
                    previous_cars: Dict[int, np.ndarray],
                    threshold: float = 2.0) -> int:
    """
    Match current cluster centroid to a previous global ID based on nearest neighbor
    within a threshold. If no match is found, allocate a new ID.
    """
    global _next_global_car_id

    closest_match = None
    min_distance = float("inf")

    for car_id, prev_centroid in previous_cars.items():
        distance = np.linalg.norm(centroid - prev_centroid)
        if distance < min_distance and distance <= threshold:
            closest_match = car_id
            min_distance = distance

    if closest_match is not None:
        return closest_match

    new_id = _next_global_car_id
    _next_global_car_id += 1
    return new_id


def is_frame_skipped(frame_number: int,
                     skip_ranges: List[Tuple[int, int]]) -> bool:
    """Return True if frame_number falls within any of the skip ranges."""
    for start, end in skip_ranges:
        if start <= frame_number <= end:
            return True
    return False


def get_candidate_previous(
    frame_number: int,
    previous_event_frames: Dict[int, Dict[int, np.ndarray]],
    previous_cars: Dict[int, np.ndarray],
    window_size: int = 3,
):
    """
    Retrieve candidate previous centroids for matching:
    - First try exact frame match.
    - Otherwise, aggregate centroids from a small window of past frames.
    """
    if frame_number in previous_event_frames:
        return previous_event_frames[frame_number], frame_number

    candidate_frames = [f for f in previous_event_frames
                        if 0 < (frame_number - f) <= window_size]
    if candidate_frames:
        candidate_mapping: Dict[int, np.ndarray] = {}
        for f in candidate_frames:
            candidate_mapping.update(previous_event_frames[f])
        closest_frame = min(candidate_frames, key=lambda f: abs(f - frame_number))
        return candidate_mapping, closest_frame

    return previous_cars, None


def is_moving(current_centroid: np.ndarray,
              previous_centroid: np.ndarray,
              frame_gap: int,
              fps: float = FPS,
              min_speed_mps: float = MIN_SPEED_MPS) -> bool:
    """
    Check whether a vehicle is moving as expected in the longitudinal (X) direction.

    Movement is approximated using delta X over frame_gap * fps.
    Negative speeds correspond to vehicles approaching the LiDAR.
    """
    dx = current_centroid[0] - previous_centroid[0]
    avg_speed_mps = (dx / frame_gap) * fps if frame_gap > 0 else dx * fps
    return avg_speed_mps < min_speed_mps


def process_frame(
    frame_number: int,
    pcd_dir: str,
    skip_ranges: List[Tuple[int, int]],
    previous_cars: Dict[int, np.ndarray],
    previous_event_frames: Dict[int, Dict[int, np.ndarray]],
    motion_window_size: int = 3,
    base_threshold: float = 2.0,
) -> Tuple[List[dict], Dict[int, np.ndarray], Dict[int, Dict[int, np.ndarray]]]:
    """
    Process a single frame:

    - Optional skip (bike stationary ranges)
    - Load and filter points
    - Segment clusters
    - Match clusters to global IDs with nearest-neighbor + motion filter
    - Return tracking records for this frame and updated previous states

    Returns
    -------
    (records, updated_previous_cars, updated_previous_event_frames)
    """
    if is_frame_skipped(frame_number, skip_ranges):
        return [], previous_cars, previous_event_frames

    points_xyz = load_point_cloud_xyz(frame_number, pcd_dir)
    if points_xyz is None:
        if VERBOSE:
            print(f"[DEBUG] Frame {frame_number}: no PCD file found.")
        return [], previous_cars, previous_event_frames

    filtered_points = filter_points(points_xyz)
    if filtered_points.size == 0:
        if VERBOSE:
            print(f"[DEBUG] Frame {frame_number}: no points after filtering.")
        return [], previous_cars, previous_event_frames

    clusters = segment_objects(filtered_points)
    if len(clusters) == 0:
        if VERBOSE:
            print(f"[DEBUG] Frame {frame_number}: no clusters detected.")
        return [], previous_cars, previous_event_frames

    candidate_previous, candidate_frame = get_candidate_previous(
        frame_number, previous_event_frames, previous_cars, window_size=motion_window_size
    )

    tracked_cars: Dict[int, dict] = {}
    temp_id_mapping: Dict[int, int] = {}

    # Match each cluster to a global ID and apply motion filter
    for cluster_id, cluster in clusters.items():
        car_id = match_global_id(cluster["centroid"],
                                 candidate_previous,
                                 threshold=base_threshold)

        if car_id in candidate_previous and candidate_frame is not None:
            frame_gap = frame_number - candidate_frame
            if not is_moving(cluster["centroid"],
                             candidate_previous[car_id],
                             frame_gap,
                             fps=FPS,
                             min_speed_mps=MIN_SPEED_MPS):
                if VERBOSE:
                    print(
                        f"[DEBUG] Frame {frame_number}: cluster {cluster_id} "
                        f"(vehicle_id {car_id}) not moving as expected, skipping."
                    )
                continue

        tracked_cars[car_id] = cluster
        temp_id_mapping[cluster_id] = car_id

    if len(tracked_cars) == 0:
        if VERBOSE:
            print(f"[DEBUG] Frame {frame_number}: no moving vehicles kept.")
        return [], previous_cars, previous_event_frames

    # Build tracking records for each accepted cluster
    frame_records: List[dict] = []
    for cluster_id, cluster in clusters.items():
        if cluster_id not in temp_id_mapping:
            continue
        car_id = temp_id_mapping[cluster_id]
        min_distance = float(np.min(np.linalg.norm(cluster["points"], axis=1)))

        frame_records.append({
            "frame": frame_number,
            "vehicle_id": car_id,
            "centroid": cluster["centroid"],
            "min_distance_m": min_distance,
            "points": cluster["points"],
        })

    updated_previous_cars = {
        car_id: cluster["centroid"] for car_id, cluster in tracked_cars.items()
    }
    updated_previous_event_frames = dict(previous_event_frames)
    updated_previous_event_frames[frame_number] = updated_previous_cars.copy()

    return frame_records, updated_previous_cars, updated_previous_event_frames
