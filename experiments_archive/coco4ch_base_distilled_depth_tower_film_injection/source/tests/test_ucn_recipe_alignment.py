from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_ucn_original_recipe_alignment() -> None:
    from baselines.run_ucn_ecc import build_ucn_recipe

    recipe = build_ucn_recipe(register="0831")

    assert recipe.input_type == "RGBD"
    assert recipe.fusion_type == "add"
    assert recipe.num_units == 64
    assert recipe.learning_rate == pytest.approx(1.0e-5)
    assert recipe.weight_decay == pytest.approx(5.0e-4)
    assert recipe.embedding_pretrain is False
    assert recipe.embedding_normalization is True
    assert recipe.embedding_metric == "cosine"
    assert recipe.embedding_alpha == pytest.approx(0.02)
    assert recipe.embedding_delta == pytest.approx(0.5)
    assert recipe.embedding_lambda_intra == pytest.approx(10.0)
    assert recipe.embedding_lambda_inter == pytest.approx(10.0)
    assert recipe.chromatic is True
    assert recipe.add_noise is True
    assert recipe.batch_size == 16
    assert recipe.num_seeds == 100
    assert recipe.kappa == pytest.approx(20.0)


def _write_min_stats_dataset(root: Path, image_size: int) -> None:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "depth" / "depth_npy" / "train").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    for split, color in (("train", (10, 20, 30)), ("val", (40, 50, 60))):
        image_name = f"{split}_000001.png"
        Image.new("RGB", (image_size, image_size), color=color).save(root / "images" / split / image_name)

    np.save(
        root / "depth" / "depth_npy" / "train" / "train_000001.npy",
        np.full((image_size, image_size), 0.5, dtype=np.float32),
    )

    payload = {
        "images": [{"id": 1, "file_name": "train_000001.png", "width": image_size, "height": image_size}],
        "annotations": [],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / "instances_train.json").write_text(json.dumps(payload), encoding="utf-8")
    payload["images"][0]["file_name"] = "val_000001.png"
    (root / "annotations" / "instances_val.json").write_text(json.dumps(payload), encoding="utf-8")


def test_ucn_custom_register_uses_dataset_root_stats(tmp_path: Path) -> None:
    from baselines.normalization_stats import load_dataset_normalization_stats
    from baselines.run_ucn_ecc import _load_depth_clip, _load_pixel_mean_bgr_255

    dataset_root = tmp_path / "20260318_1K_1566"
    _write_min_stats_dataset(dataset_root, 32)
    expected = load_dataset_normalization_stats(str(dataset_root))

    pixel_mean = _load_pixel_mean_bgr_255("20260318_1k_1566", str(dataset_root))
    depth_clip = _load_depth_clip("20260318_1k_1566", str(dataset_root))

    assert pixel_mean == pytest.approx(list(expected.rgb_mean_bgr_255))
    assert depth_clip == pytest.approx((expected.depth_clip_min, expected.depth_clip_max))


def test_ucn_main_uses_args_dataset_root_for_custom_register_stats() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "baselines" / "run_ucn_ecc.py"
    text = script.read_text(encoding="utf-8")

    assert "_load_pixel_mean_bgr_255(register_id, str(args.dataset_root))" in text
    assert "_load_depth_clip(register_id, str(args.dataset_root))" in text
