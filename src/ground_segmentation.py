"""
ground_segmentation.py

Ground segmentation utilities: binning and simple local Z-threshold ground
classification.
"""

import numpy as np

# Configuration 
BIN_SIZE = 1.0              # meters
Z_DIST_THRESHOLD = 0.2      # meters
Y_MIN, Y_MAX = -20.0, 20.0  # lateral ROI


def create_3d_bins(points: np.ndarray, bin_size: float = BIN_SIZE):
    """
    Bin points in the XY-plane using a fixed bin size, restricted to [Y_MIN, Y_MAX].

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) or (N, >=3) with XYZ in the first 3 columns.
    bin_size : float
        Bin size in meters.

    Returns
    -------
    bins : dict
        Mapping (ix, iy) -> list of point rows.
    filtered_points : np.ndarray
        Points after applying lateral Y-range filter.
    """
    y_mask = (points[:, 1] >= Y_MIN) & (points[:, 1] <= Y_MAX)
    points = points[y_mask]

    x_idx = np.floor(points[:, 0] / bin_size).astype(int)
    y_idx = np.floor(points[:, 1] / bin_size).astype(int)

    bins = {}
    for i in range(points.shape[0]):
        key = (x_idx[i], y_idx[i])
        if key not in bins:
            bins[key] = []
        bins[key].append(points[i])

    return bins, points


def classify_ground(points: np.ndarray,
                    bins: dict,
                    z_thresh: float = Z_DIST_THRESHOLD) -> np.ndarray:
    """
    Classify ground points within each XY-bin based on a local Z-threshold.

    For each bin, the minimum Z is taken as the local ground reference, and
    points whose Z is within z_thresh of this local minimum are labeled ground.

    Parameters
    ----------
    points : np.ndarray
        Points (N, 3) filtered to those used for binning.
    bins : dict
        Mapping (ix, iy) -> list of point rows (as from create_3d_bins).
    z_thresh : float
        Z distance threshold in meters.

    Returns
    -------
    ground_mask : np.ndarray (bool)
        Boolean mask of shape (N,) where True indicates ground.
    """
    ground_mask = np.zeros(len(points), dtype=bool)
    # We use exact tuple of point row as key
    index_map = {tuple(p): i for i, p in enumerate(points)}

    for pts in bins.values():
        pts = np.array(pts)
        if len(pts) == 0:
            continue
        min_z = np.min(pts[:, 2])
        close_to_ground = np.abs(pts[:, 2] - min_z) < z_thresh
        for pt in pts[close_to_ground]:
            idx = index_map.get(tuple(pt))
            if idx is not None:
                ground_mask[idx] = True

    return ground_mask
