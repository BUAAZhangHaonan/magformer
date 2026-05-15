from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "validate_derived_dataset.py"


def _encode_mask(mask: np.ndarray) -> tuple[dict[str, object], list[float], float]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    bbox = [float(value) for value in mask_utils.toBbox(encoded)]
    area = float(mask_utils.area(encoded))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}, bbox, area


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_split(root: Path, split: str) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "depth" / "depth_npy" / split).mkdir(parents=True, exist_ok=True)
    (root / "cache" / "depth_xyz" / split).mkdir(parents=True, exist_ok=True)
    (root / "cache" / "instance_map" / split).mkdir(parents=True, exist_ok=True)

    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []

    for index in range(1, 3):
        image_id = index
        ann_id = 1000 + index
        image_name = f"{split}_{index:06d}.png"
        stem = Path(image_name).stem

        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        rgb[..., 0] = 20 * index
        rgb[..., 1] = 30
        rgb[..., 2] = 40
        Image.fromarray(rgb).save(root / "images" / split / image_name)

        depth = np.arange(20, dtype=np.float32).reshape(4, 5) + index
        np.save(root / "depth" / "depth_npy" / split / f"{stem}.npy", depth)

        depth_xyz = np.dstack([depth, depth + 1.0, depth + 2.0]).astype(np.float32)
        np.save(root / "cache" / "depth_xyz" / split / f"{stem}.npy", depth_xyz)

        instance_map = np.zeros((4, 5), dtype=np.int32)
        instance_map[1:3, 2:5] = ann_id
        np.save(root / "cache" / "instance_map" / split / f"{stem}.npy", instance_map)

        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 2:5] = 1
        segmentation, bbox, area = _encode_mask(mask)

        images.append({"id": image_id, "file_name": image_name, "width": 5, "height": 4})
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": segmentation,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            }
        )

    _write_json(
        root / "annotations" / f"instances_{split}.json",
        {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "component"}],
        },
    )


def _write_dataset(root: Path, *, with_preprocess_manifest: bool = True) -> None:
    for split in ("train", "val"):
        _write_split(root, split)

    _write_json(
        root / "cache" / "derived_dataset_manifest.json",
        {"target_root": str(root.resolve()), "splits": ["train", "val"]},
    )
    rgb_stats_path = root / "cache" / "rgb_stats.json"
    depth_stats_path = root / "cache" / "depth_stats.json"
    stats_manifest_path = root / "cache" / "stats_manifest.json"
    _write_json(rgb_stats_path, {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]})
    _write_json(depth_stats_path, {"mean": 1.0, "std": 2.0})
    _write_json(stats_manifest_path, {"dataset_root": str(root.resolve())})

    if with_preprocess_manifest:
        _write_json(
            root / "cache" / "preprocess_manifest.json",
            {
                "stats_manifest_path": str(stats_manifest_path),
                "rgb_stats_path": str(rgb_stats_path),
                "depth_stats_path": str(depth_stats_path),
            },
        )


def _run_validator(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(root),
            "--sample-images-per-split",
            "2",
            "--sample-anns-per-split",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_validate_derived_dataset_outputs_summary_for_valid_dataset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "derived"
    _write_dataset(dataset_root)

    result = _run_validator(dataset_root)

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["dataset_root"] == str(dataset_root.resolve())
    assert summary["require_preprocess_manifest"] is True
    assert summary["splits"]["train"]["images"] == 2
    assert summary["splits"]["train"]["annotations"] == 2
    assert summary["splits"]["train"]["sampled_images"] == 2
    assert summary["splits"]["train"]["sampled_annotations"] == 2
    assert set(summary["splits"]) == {"train", "val"}


def test_validate_derived_dataset_fails_when_preprocess_manifest_is_missing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "derived"
    _write_dataset(dataset_root, with_preprocess_manifest=False)

    result = _run_validator(dataset_root)

    assert result.returncode != 0
    assert "cache/preprocess_manifest.json" in result.stderr
