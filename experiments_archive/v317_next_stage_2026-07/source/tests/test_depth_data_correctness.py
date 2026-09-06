from __future__ import annotations

import numpy as np
import pytest

from magformer.data.dataset import CocoRgbdDataset
from magformer.data.transforms import DepthNoiseAug, DepthNormalize, RGBDTransform


def test_depth_noise_samples_only_finite_positive_pixels(monkeypatch) -> None:
    depth = np.array(
        [[0.0, 0.4, np.nan], [0.5, np.inf, -1.0]],
        dtype=np.float32,
    )
    sampled_sizes = []

    def deterministic_normal(mean, std, size):
        sampled_sizes.append(size)
        assert mean == 0
        assert std == pytest.approx(0.1)
        return np.array([0.1, -0.1], dtype=np.float32)

    monkeypatch.setattr(np.random, "normal", deterministic_normal)

    result = DepthNoiseAug(gaussian_std=0.1)({"depth": depth})

    assert sampled_sizes == [2]
    np.testing.assert_allclose(result["depth"], [[0.0, 0.5, 0.0], [0.4, 0.0, 0.0]])


def test_depth_normalize_preserves_invalid_zero_with_nonzero_shift() -> None:
    transform = DepthNormalize(
        scale=1.0,
        shift=0.2,
        clip_min=0.0,
        clip_max=1.0,
        norm="none",
    )

    result = transform({"depth": np.array([[0.0, 0.5]], dtype=np.float32)})

    np.testing.assert_allclose(result["depth"], [[0.0, 0.7]])


def test_training_pipeline_noises_raw_valid_depth_before_normalization(monkeypatch) -> None:
    sampled_sizes = []

    def deterministic_normal(mean, std, size):
        sampled_sizes.append(size)
        return np.full(size, 0.1, dtype=np.float32)

    monkeypatch.setattr(np.random, "normal", deterministic_normal)
    transform = RGBDTransform(
        image_size=2,
        min_scale=1.0,
        max_scale=1.0,
        random_flip="none",
        depth_scale=1.0,
        depth_clip_min=0.3,
        depth_clip_max=0.7,
        depth_norm="minmax",
        depth_per_sample_norm=True,
        depth_gaussian_std=0.1,
        is_train=True,
    )

    result = transform(
        {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.array([[0.0, 0.3], [0.5, np.nan]], dtype=np.float32),
        }
    )

    assert sampled_sizes == [2]
    np.testing.assert_allclose(
        result["depth"].numpy(),
        [[[0.0, 0.25], [0.75, 0.0]]],
        atol=1e-6,
    )


def test_missing_depth_raises_with_sample_and_path_context(tmp_path) -> None:
    depth_path = tmp_path / "missing.npy"
    dataset = CocoRgbdDataset.__new__(CocoRgbdDataset)

    with pytest.raises(FileNotFoundError) as exc_info:
        dataset._load_depth(
            depth_path,
            sample_id=37,
            image_filename="train/example.png",
        )

    message = str(exc_info.value)
    assert "sample_id=37" in message
    assert "train/example.png" in message
    assert str(depth_path) in message


def test_unreadable_depth_raises_with_sample_and_path_context(tmp_path) -> None:
    depth_path = tmp_path / "corrupt.npy"
    depth_path.write_bytes(b"not a numpy array")
    dataset = CocoRgbdDataset.__new__(CocoRgbdDataset)

    with pytest.raises(RuntimeError) as exc_info:
        dataset._load_depth(
            depth_path,
            sample_id="bad-sample",
            image_filename="val/bad.png",
        )

    message = str(exc_info.value)
    assert "sample_id='bad-sample'" in message
    assert "val/bad.png" in message
    assert str(depth_path) in message
