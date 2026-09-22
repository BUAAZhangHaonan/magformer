from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from magformer.config import loader as config_loader
from magformer.data.collate import _decode_mask
from magformer.data.copy_paste_aug import CopyPasteAugmentation
from magformer.data.dataset import CocoRgbdDataset
from magformer.data.transforms import RGBDTransform


def test_config_parser_rejects_unknown_cli_args(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", "configs/base.yaml", "--typo-override", "1"],
    )

    with pytest.raises(SystemExit):
        config_loader.parse_args()


def test_setup_device_fails_when_cuda_requested_but_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        config_loader.setup_device(SimpleNamespace(device="cuda", gpus=[0]))


def test_rgbd_transform_rejects_invalid_random_flip() -> None:
    with pytest.raises(ValueError, match="random_flip"):
        RGBDTransform(random_flip="horziontal", is_train=True)


def test_dataset_rejects_three_channel_depth_arrays(tmp_path) -> None:
    depth_path = tmp_path / "bad_depth.npy"
    np.save(depth_path, np.zeros((4, 4, 3), dtype=np.float32))
    dataset = CocoRgbdDataset.__new__(CocoRgbdDataset)

    with pytest.raises(ValueError, match="single-channel"):
        dataset._load_depth(depth_path)


def test_collate_decode_mask_merges_polygon_channels_over_instance_axis() -> None:
    mask = _decode_mask([[1, 1, 3, 1, 3, 3, 1, 3]], (4, 4))

    assert tuple(mask.shape) == (4, 4)
    assert mask.dtype == torch.bool
    assert mask.any()


def test_copy_paste_training_flag_is_explicit() -> None:
    aug = CopyPasteAugmentation(training=False, prob=1.0)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    masks = [np.ones((4, 4), dtype=bool)]
    bboxes = [[0, 0, 4, 4]]
    categories = [1]

    out = aug(image, masks, bboxes, categories, current_idx=0)

    assert out == (image, masks, bboxes, categories)
