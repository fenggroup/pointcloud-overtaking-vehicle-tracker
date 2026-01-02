"""
detection.py

Low-level detection utilities:
- Load XYZ from PCD
- Ground filtering + lateral ROI
- DBSCAN clustering into object candidates
"""

import os
from typing import Dict

import numpy as np
from sklearn.cluster import DBSCAN

from ..pcd_io import load_xyz_from_pcd
from ..ground_segmentation import (
    create_3d_bins,
    classify_ground,
    BIN_SIZE,
    Z_DIST_THRESHOLD,
    Y_MIN,
    Y_MAX,
)

# ─── CONFIG ────────────────────────────────────────────────────────────────

PCD_FRAME_PATTERN = "frame-{frame:06d}.pcd"

# DBSCAN & distance thresholds (XY radial distance)
EPS = 0.85               # DBSCAN epsilon
MIN_SAMPLES = 10         # DBSCAN minimum samples
DISTANCE_THRESHOLD = 1.0 # radial distance threshold in XY plane (meters)


def load_point_cloud_xyz(frame_number: int,
                         pcd_dir: str) -> np.ndarray:
    """
    Load a single frame's PCD as XYZ array.

    Returns None if file does not exist.
    """
    frame_file = os.path.join(pcd_dir, PCD_FRAME_PATTERN.format(frame=frame_number))
    if not os.path.exists(frame_file):
        return None
    points = load_xyz_from_pcd(frame_file)
    return points


def filter_points(points: np.ndarray) -> np.ndarray:
    """
    Apply lateral Y filter, ground removal, and keep non-ground points
    belonging to objects of interest (e.g., vehicles on the left).
    """
    # Apply lateral Y-range (consistent with ground_segmentation)
    y_mask = (points[:, 1] >= Y_MIN) & (points[:, 1] <= Y_MAX)
    points_yfiltered = points[y_mask]

    # Bin and classify ground
    bins, filtered_points = create_3d_bins(points_yfiltered, BIN_SIZE)
    ground_mask = classify_ground(filtered_points, bins, Z_DIST_THRESHOLD)
    non_ground_points = filtered_points[~ground_mask]

    # Keep only points at minimum radial distance and on the left side (y <= -1.0)
    mask = (
        (np.linalg.norm(non_ground_points[:, :2], axis=1) > DISTANCE_THRESHOLD) &
        (non_ground_points[:, 1] <= -1.0)
    )
    return non_ground_points[mask]


def segment_objects(points: np.ndarray) -> Dict[int, dict]:
    """
    Segment objects using DBSCAN and compute centroids.

    Returns
    -------
    dict
        Mapping cluster_label -> {"points": cluster_points, "centroid": centroid_xyz}
    """
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
    labels = dbscan.fit_predict(points)

    clusters: Dict[int, dict] = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = points[labels == label]
        if cluster_points.size == 0:
            continue
        centroid = np.mean(cluster_points, axis=0)
        # Skip tiny clusters right at the LiDAR
        if np.linalg.norm(centroid) < 0.5:
            continue
        clusters[label] = {"points": cluster_points, "centroid": centroid}
    return clusters
