from __future__ import annotations

import numpy as np


def centered_camera_params(*, height: int, width: int) -> dict[str, float]:
    focal = max(float(max(int(height), int(width)) - 1), 1.0)
    return {
        "fx": focal,
        "fy": focal,
        "x_offset": (float(width) - 1.0) * 0.5,
        "y_offset": (float(height) - 1.0) * 0.5,
    }


def depth_to_xyz(depth: np.ndarray, camera_params: dict[str, float] | None = None) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)

    if depth.ndim == 3 and depth.shape[2] == 3:
        return depth
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected scalar depth map or 3-channel geometry, got shape={depth.shape}")

    height, width = depth.shape
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    params = camera_params or centered_camera_params(height=height, width=width)
    fx = float(params["fx"])
    fy = float(params["fy"])
    cx = float(params["x_offset"])
    cy = float(params["y_offset"])

    x = ((xs[None, :] - cx) / max(fx, 1.0)) * depth
    y = ((ys[:, None] - cy) / max(fy, 1.0)) * depth
    z = depth
    return np.stack([x, y, z], axis=2).astype(np.float32, copy=False)


def depth_to_xyz_like(depth: np.ndarray) -> np.ndarray:
    """
    Convert scalar depth into a simple 3-channel geometry encoding.

    Upstream UCN/MSMFormer RGBD backbones expect ordered XYZ-like geometry rather
    than the same scalar depth repeated three times. ECC depth exports do not ship
    camera intrinsics alongside each sample, so we approximate XYZ with normalized
    image coordinates scaled by depth:

      x = ((u - cx) / fx) * z
      y = ((v - cy) / fy) * z
      z = depth

    If depth is already 3-channel, keep it unchanged.
    """

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[2] == 3:
        return depth
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected scalar depth map or 3-channel geometry, got shape={depth.shape}")

    return depth_to_xyz(depth)
