#!/usr/bin/env python3
"""Strict read-only validation for derived MAGFormer RGB-D COCO datasets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from pycocotools import mask as mask_utils


SPLITS = ("train", "val", "test")
REQUIRED_CATEGORY_ID = 1
REQUIRED_CATEGORY_NAME = "component"
TOLERANCE = 1e-5


class ValidationError(RuntimeError):
    """Raised when a dataset invariant is violated."""


def _display(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _load_json(path: Path, root: Path) -> Any:
    if not path.is_file():
        raise ValidationError(f"Missing required file: {_display(path, root)}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {_display(path, root)}: {exc}") from exc


def _require_dict(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{context} must be a JSON object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{context} must be a JSON array")
    return value


def _require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{context} must be an integer")
    return value


def _require_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{context} must be finite")
    return number


def _require_rel_path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{context} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValidationError(f"{context} must stay inside its split directory: {value}")
    return path


def _require_dir(path: Path, root: Path) -> None:
    if not path.is_dir():
        raise ValidationError(f"Missing required directory: {_display(path, root)}")


def _require_file(path: Path, root: Path) -> None:
    if not path.is_file():
        raise ValidationError(f"Missing required file: {_display(path, root)}")


def _count_files(path: Path, root: Path) -> int:
    _require_dir(path, root)
    return sum(1 for child in path.iterdir() if child.is_file())


def _sample_rows(rows: list[Any], limit: int, key: str) -> list[Any]:
    if limit <= 0:
        return []
    keyed: list[tuple[str, int, Any]] = []
    for idx, row in enumerate(rows):
        value = row.get(key) if isinstance(row, dict) else None
        keyed.append((str(value), idx, row))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in keyed[:limit]]


def _load_npy(path: Path, root: Path, context: str) -> np.ndarray:
    _require_file(path, root)
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - loader errors are validation failures.
        raise ValidationError(f"{context} failed to load {_display(path, root)}: {exc}") from exc


def _index_images(images: list[Any], split: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for offset, raw_image in enumerate(images):
        context = f"annotations/instances_{split}.json images[{offset}]"
        image = _require_dict(raw_image, context)
        image_id = _require_int(image.get("id"), f"{context}.id")
        if image_id in indexed:
            raise ValidationError(f"{context}.id duplicates image id {image_id}")
        _require_int(image.get("width"), f"{context}.width")
        _require_int(image.get("height"), f"{context}.height")
        _require_rel_path(image.get("file_name"), f"{context}.file_name")
        indexed[image_id] = image
    return indexed


def _assert_component_category(categories: list[Any], split: str) -> None:
    for raw_category in categories:
        if not isinstance(raw_category, dict):
            continue
        if raw_category.get("id") == REQUIRED_CATEGORY_ID and raw_category.get("name") == REQUIRED_CATEGORY_NAME:
            return
    raise ValidationError(
        f"annotations/instances_{split}.json categories must include "
        f"id={REQUIRED_CATEGORY_ID}, name={REQUIRED_CATEGORY_NAME!r}"
    )


def _validate_image_sample(dataset_root: Path, split: str, raw_image: Any) -> None:
    image = _require_dict(raw_image, f"annotations/instances_{split}.json sampled image")
    image_id = _require_int(image.get("id"), f"split {split} image.id")
    width = _require_int(image.get("width"), f"split {split} image {image_id}.width")
    height = _require_int(image.get("height"), f"split {split} image {image_id}.height")
    rel_image = _require_rel_path(image.get("file_name"), f"split {split} image {image_id}.file_name")
    rel_npy = rel_image.with_suffix(".npy")

    rgb_path = dataset_root / "images" / split / rel_image
    _require_file(rgb_path, dataset_root)
    try:
        with Image.open(rgb_path) as rgb:
            if rgb.size != (width, height):
                raise ValidationError(
                    f"split {split} image {image_id} RGB size {rgb.size} != JSON size {(width, height)}"
                )
    except UnidentifiedImageError as exc:
        raise ValidationError(f"split {split} image {image_id} is not a readable image: {exc}") from exc
    except OSError as exc:
        raise ValidationError(f"split {split} image {image_id} failed to read RGB image: {exc}") from exc

    depth = _load_npy(
        dataset_root / "depth" / "depth_npy" / split / rel_npy,
        dataset_root,
        f"split {split} image {image_id} depth",
    )
    if depth.shape != (height, width):
        raise ValidationError(f"split {split} image {image_id} depth shape {depth.shape} != {(height, width)}")

    depth_xyz = _load_npy(
        dataset_root / "cache" / "depth_xyz" / split / rel_npy,
        dataset_root,
        f"split {split} image {image_id} depth_xyz",
    )
    if depth_xyz.shape != (height, width, 3):
        raise ValidationError(f"split {split} image {image_id} depth_xyz shape {depth_xyz.shape} != {(height, width, 3)}")

    instance_map = _load_npy(
        dataset_root / "cache" / "instance_map" / split / rel_npy,
        dataset_root,
        f"split {split} image {image_id} instance_map",
    )
    if instance_map.shape != (height, width):
        raise ValidationError(
            f"split {split} image {image_id} instance_map shape {instance_map.shape} != {(height, width)}"
        )


def _decode_rle(segmentation: Any, height: int, width: int, context: str) -> np.ndarray:
    if not isinstance(segmentation, dict):
        raise ValidationError(f"{context}.segmentation must be RLE with size/counts")
    size = segmentation.get("size")
    if size != [height, width]:
        raise ValidationError(f"{context}.segmentation.size {size!r} != {[height, width]}")
    if "counts" not in segmentation:
        raise ValidationError(f"{context}.segmentation missing counts")

    counts = segmentation["counts"]
    if isinstance(counts, str):
        rle = dict(segmentation)
        rle["counts"] = counts.encode("ascii")
    elif isinstance(counts, bytes):
        rle = dict(segmentation)
    elif isinstance(counts, list):
        try:
            rle = mask_utils.frPyObjects(segmentation, height, width)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(f"{context}.segmentation cannot be converted to RLE: {exc}") from exc
    else:
        raise ValidationError(f"{context}.segmentation counts must be string, bytes, or list")

    try:
        mask = mask_utils.decode(rle)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"{context}.segmentation cannot be decoded: {exc}") from exc
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.shape != (height, width):
        raise ValidationError(f"{context} decoded mask shape {mask.shape} != {(height, width)}")
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        raise ValidationError(f"{context} decoded mask is empty")
    return mask_bool


def _mask_bbox(mask: np.ndarray) -> list[float]:
    rows, cols = np.where(mask)
    x_min = int(cols.min())
    y_min = int(rows.min())
    x_max = int(cols.max())
    y_max = int(rows.max())
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def _validate_annotation_sample(split: str, raw_ann: Any, image_by_id: dict[int, dict[str, Any]]) -> None:
    ann = _require_dict(raw_ann, f"annotations/instances_{split}.json sampled annotation")
    ann_id = ann.get("id", "<missing>")
    context = f"split {split} annotation {ann_id}"
    image_id = _require_int(ann.get("image_id"), f"{context}.image_id")
    if image_id not in image_by_id:
        raise ValidationError(f"{context} references missing image_id {image_id}")
    image = image_by_id[image_id]
    width = _require_int(image.get("width"), f"{context} image.width")
    height = _require_int(image.get("height"), f"{context} image.height")
    if "segmentation" not in ann:
        raise ValidationError(f"{context} missing segmentation")
    mask = _decode_rle(ann["segmentation"], height, width, context)

    bbox = _require_list(ann.get("bbox"), f"{context}.bbox")
    if len(bbox) != 4:
        raise ValidationError(f"{context}.bbox must contain 4 values")
    bbox_values = [_require_number(value, f"{context}.bbox[{idx}]") for idx, value in enumerate(bbox)]
    expected_bbox = _mask_bbox(mask)
    for idx, (actual, expected) in enumerate(zip(bbox_values, expected_bbox)):
        if abs(actual - expected) > TOLERANCE:
            raise ValidationError(f"{context}.bbox[{idx}] {actual} != decoded mask bbox {expected}")

    area = _require_number(ann.get("area"), f"{context}.area")
    expected_area = float(mask.sum())
    if abs(area - expected_area) > TOLERANCE:
        raise ValidationError(f"{context}.area {area} != decoded mask area {expected_area}")


def _validate_split(
    dataset_root: Path,
    split: str,
    sample_images_per_split: int,
    sample_anns_per_split: int,
) -> dict[str, int]:
    ann_path = dataset_root / "annotations" / f"instances_{split}.json"
    payload = _require_dict(_load_json(ann_path, dataset_root), _display(ann_path, dataset_root))
    images = _require_list(payload.get("images"), f"annotations/instances_{split}.json images")
    annotations = _require_list(payload.get("annotations"), f"annotations/instances_{split}.json annotations")
    categories = _require_list(payload.get("categories"), f"annotations/instances_{split}.json categories")
    _assert_component_category(categories, split)
    image_by_id = _index_images(images, split)

    expected_count = len(images)
    for label, path in {
        "images": dataset_root / "images" / split,
        "depth_npy": dataset_root / "depth" / "depth_npy" / split,
        "depth_xyz": dataset_root / "cache" / "depth_xyz" / split,
        "instance_map": dataset_root / "cache" / "instance_map" / split,
    }.items():
        count = _count_files(path, dataset_root)
        if count != expected_count:
            raise ValidationError(
                f"split {split} {label} file count {count} != annotations image count {expected_count}"
            )

    sampled_images = _sample_rows(images, sample_images_per_split, "file_name")
    for image in sampled_images:
        _validate_image_sample(dataset_root, split, image)

    sampled_annotations = _sample_rows(annotations, sample_anns_per_split, "id")
    for ann in sampled_annotations:
        _validate_annotation_sample(split, ann, image_by_id)

    return {
        "images": expected_count,
        "annotations": len(annotations),
        "sampled_images": len(sampled_images),
        "sampled_annotations": len(sampled_annotations),
    }


def _discover_splits(dataset_root: Path) -> list[str]:
    ann_dir = dataset_root / "annotations"
    if not ann_dir.is_dir():
        raise ValidationError(f"Missing required directory: {_display(ann_dir, dataset_root)}")
    splits = [split for split in SPLITS if (ann_dir / f"instances_{split}.json").is_file()]
    if not splits:
        raise ValidationError(f"No instances_<split>.json files found under {_display(ann_dir, dataset_root)}")
    return splits


def _validate_preprocess_manifest(dataset_root: Path) -> dict[str, str]:
    derived_path = dataset_root / "cache" / "derived_dataset_manifest.json"
    preprocess_path = dataset_root / "cache" / "preprocess_manifest.json"
    _load_json(derived_path, dataset_root)
    preprocess = _require_dict(_load_json(preprocess_path, dataset_root), "cache/preprocess_manifest.json")

    refs: dict[str, str] = {}
    for key in ("stats_manifest_path", "rgb_stats_path", "depth_stats_path"):
        value = preprocess.get(key)
        if not isinstance(value, str) or not value:
            raise ValidationError(f"cache/preprocess_manifest.json {key} must be a non-empty path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = dataset_root / path
        path = path.resolve()
        if not path.is_file():
            raise ValidationError(f"Missing required file referenced by {key}: {path}")
        refs[key] = str(path)

    stats_manifest = _require_dict(_load_json(Path(refs["stats_manifest_path"]), dataset_root), refs["stats_manifest_path"])
    stats_dataset_root = stats_manifest.get("dataset_root")
    if not isinstance(stats_dataset_root, str) or not stats_dataset_root:
        raise ValidationError("stats manifest dataset_root must be a non-empty path")
    if Path(stats_dataset_root).resolve() != dataset_root.resolve():
        raise ValidationError(
            f"stats manifest dataset_root {Path(stats_dataset_root).resolve()} != {dataset_root.resolve()}"
        )
    return refs


def validate_dataset(
    dataset_root: Path,
    sample_images_per_split: int,
    sample_anns_per_split: int,
    require_preprocess_manifest: bool,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise ValidationError(f"Dataset root does not exist: {dataset_root}")
    splits = _discover_splits(dataset_root)
    split_summaries = {
        split: _validate_split(dataset_root, split, sample_images_per_split, sample_anns_per_split)
        for split in splits
    }
    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "require_preprocess_manifest": bool(require_preprocess_manifest),
        "splits": split_summaries,
    }
    if require_preprocess_manifest:
        summary["preprocess_manifest"] = _validate_preprocess_manifest(dataset_root)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a derived MAGFormer RGB-D COCO dataset before training.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sample-images-per-split", type=int, default=5)
    parser.add_argument("--sample-anns-per-split", type=int, default=30)
    parser.add_argument(
        "--require-preprocess-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require cache/preprocess_manifest.json and referenced stats files. Use --no-require-preprocess-manifest to disable.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = validate_dataset(
            dataset_root=args.dataset_root,
            sample_images_per_split=int(args.sample_images_per_split),
            sample_anns_per_split=int(args.sample_anns_per_split),
            require_preprocess_manifest=bool(args.require_preprocess_manifest),
        )
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
