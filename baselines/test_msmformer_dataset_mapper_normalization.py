from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _make_cfg(use_other_backbone: bool) -> SimpleNamespace:
    return SimpleNamespace(
        INPUT=SimpleNamespace(FORMAT="BGR", MASK_FORMAT="bitmask", MIN_SIZE_TEST=2),
        MODEL=SimpleNamespace(
            PIXEL_MEAN=[10.0, 20.0, 30.0],
            PIXEL_STD=[2.0, 4.0, 5.0],
            USE_OTHER_BACKBONE=use_other_backbone,
        ),
    )


def test_msmformer_mapper_uses_ucn_mean_only_rgb_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    from baselines import run_msmformer_ecc as module

    module._DEPTH_MIN = 0.0
    module._DEPTH_MAX = 1.0

    rgb = np.array(
        [
            [[10.0, 20.0, 30.0], [110.0, 120.0, 130.0]],
            [[210.0, 220.0, 230.0], [40.0, 50.0, 60.0]],
        ],
        dtype=np.float32,
    )
    depth = np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr(module.d2_utils, "read_image", lambda *_args, **_kwargs: rgb.copy())
    monkeypatch.setattr(module.np, "load", lambda *_args, **_kwargs: depth.copy())

    mapper = module.DatasetMapperRGBD(_make_cfg(use_other_backbone=False), is_train=False)
    out = mapper({"file_name": "dummy.png", "depth_file_name": "dummy.npy"})

    image = out["image"].permute(1, 2, 0).numpy()
    expected = (rgb - np.array([10.0, 20.0, 30.0], dtype=np.float32)) / 255.0
    assert np.allclose(image, expected, atol=1e-6)


def test_msmformer_mapper_keeps_std_normalization_for_other_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    from baselines import run_msmformer_ecc as module

    module._DEPTH_MIN = 0.0
    module._DEPTH_MAX = 1.0

    rgb = np.array(
        [
            [[10.0, 20.0, 30.0], [110.0, 120.0, 130.0]],
            [[210.0, 220.0, 230.0], [40.0, 50.0, 60.0]],
        ],
        dtype=np.float32,
    )
    depth = np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr(module.d2_utils, "read_image", lambda *_args, **_kwargs: rgb.copy())
    monkeypatch.setattr(module.np, "load", lambda *_args, **_kwargs: depth.copy())

    mapper = module.DatasetMapperRGBD(_make_cfg(use_other_backbone=True), is_train=False)
    out = mapper({"file_name": "dummy.png", "depth_file_name": "dummy.npy"})

    image = out["image"].permute(1, 2, 0).numpy()
    expected = (rgb - np.array([10.0, 20.0, 30.0], dtype=np.float32)) / np.array([2.0, 4.0, 5.0], dtype=np.float32)
    assert np.allclose(image, expected, atol=1e-6)


def test_msmformer_mapper_projects_depth_to_xyz_for_ucn_route(monkeypatch: pytest.MonkeyPatch) -> None:
    from baselines import run_msmformer_ecc as module
    from baselines.rgbd_geometry import centered_camera_params, depth_to_xyz

    module._DEPTH_MIN = 0.0
    module._DEPTH_MAX = 1.0

    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    depth = np.array(
        [
            [0.0, 0.25],
            [0.5, 1.0],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(module.d2_utils, "read_image", lambda *_args, **_kwargs: rgb.copy())
    monkeypatch.setattr(module.np, "load", lambda *_args, **_kwargs: depth.copy())

    mapper = module.DatasetMapperRGBD(_make_cfg(use_other_backbone=False), is_train=False)
    out = mapper({"file_name": "dummy.png", "depth_file_name": "dummy.npy"})

    depth_out = out["depth"].permute(1, 2, 0).numpy()
    expected = depth_to_xyz(depth, centered_camera_params(height=2, width=2))
    assert np.allclose(depth_out, expected, atol=1e-6)


def test_msmformer_mapper_converts_scalar_depth_to_xyz_like_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    from baselines import run_msmformer_ecc as module

    module._DEPTH_MIN = 1.0
    module._DEPTH_MAX = 4.0

    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    depth = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(module.d2_utils, "read_image", lambda *_args, **_kwargs: rgb.copy())
    monkeypatch.setattr(module.np, "load", lambda *_args, **_kwargs: depth.copy())

    mapper = module.DatasetMapperRGBD(_make_cfg(use_other_backbone=False), is_train=False)
    out = mapper({"file_name": "dummy.png", "depth_file_name": "dummy.npy"})

    depth_tensor = out["depth"].permute(1, 2, 0).numpy()
    assert depth_tensor.shape == (2, 2, 3)
    assert depth_tensor[0, 0, 0] <= 0.0
    assert depth_tensor[0, 1, 0] > 0.0
    assert depth_tensor[1, 0, 0] < 0.0
    assert depth_tensor[1, 1, 0] > 0.0
    assert depth_tensor[0, 0, 1] <= 0.0
    assert depth_tensor[0, 1, 1] < 0.0
    assert depth_tensor[1, 0, 1] > 0.0
    assert depth_tensor[1, 1, 1] > 0.0
    assert depth_tensor[0, 0, 2] < depth_tensor[0, 1, 2] < depth_tensor[1, 0, 2] < depth_tensor[1, 1, 2]
    assert not np.allclose(depth_tensor[:, :, 0], depth_tensor[:, :, 2])
    assert not np.allclose(depth_tensor[:, :, 1], depth_tensor[:, :, 2])


def test_msmformer_mapper_emits_instance_id_label_map(monkeypatch: pytest.MonkeyPatch) -> None:
    from baselines import run_msmformer_ecc as module

    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    depth = np.ones((2, 2), dtype=np.float32)

    class _Masks:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

    class _Instances:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.gt_masks = _Masks(tensor)
            self.gt_classes = torch.tensor([0, 0], dtype=torch.int64)

    mask_tensor = torch.tensor(
        [
            [[1, 0], [0, 0]],
            [[0, 1], [1, 0]],
        ],
        dtype=torch.uint8,
    )

    monkeypatch.setattr(module.d2_utils, "read_image", lambda *_args, **_kwargs: rgb.copy())
    monkeypatch.setattr(module.np, "load", lambda *_args, **_kwargs: depth.copy())
    monkeypatch.setattr(module.d2_utils, "transform_instance_annotations", lambda obj, *_args, **_kwargs: obj)
    monkeypatch.setattr(module.d2_utils, "annotations_to_instances", lambda *_args, **_kwargs: _Instances(mask_tensor.clone()))
    monkeypatch.setattr(module.d2_utils, "filter_empty_instances", lambda instances: instances)

    mapper = module.DatasetMapperRGBD(_make_cfg(use_other_backbone=False), is_train=False)
    out = mapper(
        {
            "file_name": "dummy.png",
            "depth_file_name": "dummy.npy",
            "annotations": [{}, {}],
        }
    )

    expected = torch.tensor([[0, 1], [1, -1]], dtype=torch.int64)
    assert torch.equal(out["label"].squeeze(0), expected)
