"""
pcd_io.py

Utilities for reading full-channel Ouster PCD files while exposing
simple XYZ arrays or Open3D point clouds.
"""

import numpy as np
from pypcd4 import PointCloud
import open3d as o3d


def load_xyz_from_pcd(path: str) -> np.ndarray:
    """
    Load a PCD file that contains multiple Ouster channels and return an (N, 3)
    numpy array of [x, y, z] coordinates.

    Parameters
    ----------
    path : str
        Path to the .pcd file.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with columns [x, y, z].
    """

    
    pc = PointCloud.from_path(path)
    xyz = pc.numpy(("x", "y", "z"))     # shape (N, 3)

    return xyz.astype(np.float32)


def load_o3d_from_pcd(path: str) -> o3d.geometry.PointCloud:
    """
    Load a PCD file as an Open3D point cloud, using pypcd4 for robust parsing
    and then converting to an Open3D PointCloud object.

    Parameters
    ----------
    path : str
        Path to the .pcd file.

    Returns
    -------
    o3d.geometry.PointCloud
        Open3D point cloud with only XYZ coordinates.
    """
    xyz = load_xyz_from_pcd(path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd
