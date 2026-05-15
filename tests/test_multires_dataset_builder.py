from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def _encode_mask(mask: np.ndarray) -> dict[str, object]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "size": list(encoded["size"]),
        "counts": counts,
    }


def _encode_uncompressed_mask(mask: np.ndarray) -> dict[str, object]:
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    counts: list[int] = []
    last = 0
    run = 0
    for value in flat:
        value = int(value)
        if value == last:
            run += 1
        else:
            counts.append(run)
            run = 1
            last = value
    counts.append(run)
    return {
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "counts": counts,
    }


def _write_min_rgbd_split(root: Path, split: str) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "depth" / "depth_npy" / split).mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    image_name = f"{split}_000001.png"
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 20
    rgb[..., 1] = 40
    rgb[..., 2] = 80
    Image.fromarray(rgb, mode="RGB").save(root / "images" / split / image_name)

    depth = np.linspace(0.1, 0.9, 16, dtype=np.float32).reshape(4, 4)
    np.save(root / "depth" / "depth_npy" / split / f"{split}_000001.npy", depth)

    compressed_mask = np.zeros((4, 4), dtype=np.uint8)
    compressed_mask[0:2, 0:2] = 1
    uncompressed_mask = np.zeros((4, 4), dtype=np.uint8)
    uncompressed_mask[0:2, 2:4] = 1
    polygon = [[2.0, 2.0, 4.0, 2.0, 4.0, 4.0, 2.0, 4.0]]
    ann = {
        "images": [{"id": 1, "file_name": image_name, "width": 4, "height": 4}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": _encode_mask(compressed_mask),
                "bbox": [0.0, 0.0, 2.0, 2.0],
                "area": 4.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "segmentation": _encode_uncompressed_mask(uncompressed_mask),
                "bbox": [2.0, 0.0, 2.0, 2.0],
                "area": 4.0,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 1,
                "category_id": 1,
                "segmentation": polygon,
                "bbox": [2.0, 2.0, 2.0, 2.0],
                "area": 4.0,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(ann), encoding="utf-8")


def _write_source_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        _write_min_rgbd_split(root, split)
    (root / "dataset_info.json").write_text(json.dumps({"camera": {"width": 4, "height": 4}}), encoding="utf-8")
    (root / "build_stats.json").write_text(json.dumps({"images_written": 3}), encoding="utf-8")


def test_build_multires_dataset_materializes_resized_assets_and_caches(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "build_multires_dataset.py"
    stats_script = repo_root / "scripts" / "analysis" / "ensure_dataset_stats.py"

    source_root = tmp_path / "20260318_1K_1566"
    target_root = tmp_path / "20260318_1K_1566_2"
    _write_source_dataset(source_root)

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-root",
            str(source_root),
            "--target-root",
            str(target_root),
            "--image-size",
            "2",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    resized = Image.open(target_root / "images" / "train" / "train_000001.png")
    depth = np.load(target_root / "depth" / "depth_npy" / "train" / "train_000001.npy")
    instance_map = np.load(target_root / "cache" / "instance_map" / "train" / "train_000001.npy")
    depth_xyz = np.load(target_root / "cache" / "depth_xyz" / "train" / "train_000001.npy")

    assert resized.size == (2, 2)
    assert depth.shape == (2, 2)
    assert instance_map.shape == (2, 2)
    assert int(instance_map.max()) >= 1
    assert depth_xyz.shape == (2, 2, 3)

    payload = json.loads((target_root / "cache" / "derived_dataset_manifest.json").read_text(encoding="utf-8"))
    preprocess_manifest = json.loads((target_root / "cache" / "preprocess_manifest.json").read_text(encoding="utf-8"))
    ann = json.loads((target_root / "annotations" / "instances_train.json").read_text(encoding="utf-8"))

    assert payload["source_root"] == str(source_root.resolve())
    assert payload["image_size"] == 2
    assert preprocess_manifest["instance_map_dir"].endswith("cache/instance_map")
    assert preprocess_manifest["depth_xyz_dir"].endswith("cache/depth_xyz")
    assert ann["images"][0]["width"] == 2
    assert ann["images"][0]["height"] == 2
    assert len(ann["annotations"]) == 3
    for resized_ann in ann["annotations"]:
        assert resized_ann["area"] > 0.0
        x, y, width, height = resized_ann["bbox"]
        assert 0.0 <= x <= 2.0
        assert 0.0 <= y <= 2.0
        assert width > 0.0
        assert height > 0.0
        assert x + width <= 2.0
        assert y + height <= 2.0
        assert resized_ann["segmentation"]["size"] == [2, 2]
        decoded = mask_utils.decode(resized_ann["segmentation"])
        assert decoded.shape == (2, 2)
        assert int(decoded.max()) == 1

    source_stats = json.loads(
        subprocess.run(
            [sys.executable, str(stats_script), "--dataset-root", str(source_root)],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    target_stats = json.loads(
        subprocess.run(
            [sys.executable, str(stats_script), "--dataset-root", str(target_root)],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )

    assert source_stats["cache_dir"] != target_stats["cache_dir"]
    assert Path(target_stats["rgb_stats_path"]).exists()
    assert Path(target_stats["depth_stats_path"]).exists()
