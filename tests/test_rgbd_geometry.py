from __future__ import annotations

import numpy as np
import pytest


def test_centered_camera_params_use_image_center_and_square_focal() -> None:
    from baselines.rgbd_geometry import centered_camera_params

    params = centered_camera_params(height=4, width=6)

    assert params["fx"] == pytest.approx(5.0)
    assert params["fy"] == pytest.approx(5.0)
    assert params["x_offset"] == pytest.approx(2.5)
    assert params["y_offset"] == pytest.approx(1.5)


def test_depth_to_xyz_matches_ucn_geometry_projection() -> None:
    from baselines.rgbd_geometry import centered_camera_params, depth_to_xyz

    depth = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    xyz = depth_to_xyz(depth, centered_camera_params(height=2, width=2))
    expected = np.array(
        [
            [[-0.5, -0.5, 1.0], [1.0, -1.0, 2.0]],
            [[-1.5, 1.5, 3.0], [2.0, 2.0, 4.0]],
        ],
        dtype=np.float32,
    )
    assert np.allclose(xyz, expected, atol=1.0e-6)
