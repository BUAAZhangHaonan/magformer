from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "build_resized_coco_rgbd_dataset.py"


def _encode_mask(mask: np.ndarray) -> dict[str, object]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": list(encoded["size"]), "counts": counts}


def _write_rgbd_source(root: Path, *, include_depth: bool = True) -> Path:
    split = "train"
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "depth" / "depth_npy" / split).mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(4, dtype=np.uint8)[None, :] * 30
    rgb[..., 1] = np.arange(4, dtype=np.uint8)[:, None] * 40
    rgb[..., 2] = 80
    Image.fromarray(rgb, mode="RGB").save(root / "images" / split / "sample.png")

    if include_depth:
        depth = np.arange(16, dtype=np.float32).reshape(4, 4) + 0.25
        np.save(root / "depth" / "depth_npy" / split / "sample.npy", depth)

    rle_mask = np.zeros((4, 4), dtype=np.uint8)
    rle_mask[0:2, 0:2] = 1

    ann = {
        "info": {"description": "synthetic"},
        "licenses": [{"id": 1, "name": "test"}],
        "images": [{"id": 7, "file_name": "sample.png", "width": 4, "height": 4}],
        "annotations": [
            {
                "id": 11,
                "image_id": 7,
                "category_id": 1,
                "segmentation": [[2.0, 2.0, 4.0, 2.0, 4.0, 4.0, 2.0, 4.0]],
                "bbox": [2.0, 2.0, 2.0, 2.0],
                "area": 4.0,
                "iscrowd": 0,
            },
            {
                "id": 12,
                "image_id": 7,
                "category_id": 1,
                "segmentation": _encode_mask(rle_mask),
                "bbox": [0.0, 0.0, 2.0, 2.0],
                "area": 4.0,
                "iscrowd": 1,
            },
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    ann_path = root / "annotations" / "instances_train.json"
    ann_path.write_text(json.dumps(ann), encoding="utf-8")
    return ann_path


def _run_builder(source_root: Path, output_root: Path, summary_json: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source_root),
            "--source-ann",
            "annotations/instances_train.json",
            "--output-root",
            str(output_root),
            "--output-ann",
            "annotations/instances_train_scale05.json",
            "--scale",
            "0.5",
            "--max-images",
            "1",
            "--split",
            "train",
            "--summary-json",
            str(summary_json),
            *extra,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_resizes_rgb_depth_and_scales_polygon_and_rle_annotations(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    summary_json = tmp_path / "summary.json"
    _write_rgbd_source(source_root)

    result = _run_builder(source_root, output_root, summary_json)

    assert result.returncode == 0, result.stderr
    rgb = Image.open(output_root / "images" / "train" / "sample.png")
    depth = np.load(output_root / "depth" / "depth_npy" / "train" / "sample.npy")
    out_ann = json.loads((output_root / "annotations" / "instances_train_scale05.json").read_text(encoding="utf-8"))
    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    assert rgb.size == (2, 2)
    assert depth.shape == (2, 2)
    assert depth.dtype == np.float32
    assert out_ann["images"] == [{"id": 7, "file_name": "sample.png", "width": 2, "height": 2}]
    assert out_ann["categories"] == [{"id": 1, "name": "component"}]
    assert out_ann["info"] == {"description": "synthetic"}
    assert out_ann["licenses"] == [{"id": 1, "name": "test"}]

    polygon_ann = next(ann for ann in out_ann["annotations"] if ann["id"] == 11)
    assert polygon_ann["bbox"] == [1.0, 1.0, 1.0, 1.0]
    assert polygon_ann["area"] == 1.0
    assert polygon_ann["segmentation"] == [[1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0]]

    rle_ann = next(ann for ann in out_ann["annotations"] if ann["id"] == 12)
    assert rle_ann["segmentation"]["size"] == [2, 2]
    decoded = mask_utils.decode({**rle_ann["segmentation"], "counts": rle_ann["segmentation"]["counts"].encode("ascii")})
    assert decoded.shape == (2, 2)
    assert int(decoded.sum()) == 1
    assert rle_ann["bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert rle_ann["area"] == 1.0

    assert summary["images"] == 1
    assert summary["annotations"] == 2
    assert summary["mask_area_p50"] == 1.0
    assert summary["area_le_256_ratio"] == 1.0
    assert summary["file_existence_counts"] == {"rgb": 1, "depth_npy": 1}


def test_refuses_to_overwrite_without_overwrite_flag(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    summary_json = tmp_path / "summary.json"
    _write_rgbd_source(source_root)

    first = _run_builder(source_root, output_root, summary_json)
    second = _run_builder(source_root, output_root, summary_json)

    assert first.returncode == 0, first.stderr
    assert second.returncode != 0
    assert "already exists" in second.stderr


def test_overwrite_flag_replaces_existing_output(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    summary_json = tmp_path / "summary.json"
    _write_rgbd_source(source_root)

    first = _run_builder(source_root, output_root, summary_json)
    marker = output_root / "stale.txt"
    marker.write_text("stale", encoding="utf-8")
    second = _run_builder(source_root, output_root, summary_json, "--overwrite")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert not marker.exists()


def test_missing_depth_fails_loudly(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    summary_json = tmp_path / "summary.json"
    _write_rgbd_source(source_root, include_depth=False)

    result = _run_builder(source_root, output_root, summary_json)

    assert result.returncode != 0
    assert "Depth file not found" in result.stderr
