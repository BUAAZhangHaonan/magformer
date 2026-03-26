#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.rgbd_geometry import depth_to_xyz_like
from scripts.analysis.ensure_dataset_stats import ensure_dataset_stats


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any] | List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Target root already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _iter_splits(source_root: Path) -> List[str]:
    ann_dir = source_root / "annotations"
    splits: List[str] = []
    for split in ("train", "val", "test"):
        if (ann_dir / f"instances_{split}.json").exists():
            splits.append(split)
    if not splits:
        raise FileNotFoundError(f"No instances_<split>.json files found under {ann_dir}")
    return splits


def _decode_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
    elif isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)!r}")
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2).astype(np.uint8)
    return mask.astype(np.uint8, copy=False)


def _encode_mask(mask: np.ndarray) -> Dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "size": list(encoded["size"]),
        "counts": counts,
    }


def _mask_bbox_and_area(mask: np.ndarray) -> tuple[list[float], float]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    bbox = mask_utils.toBbox(encoded).tolist()
    area = float(mask_utils.area(encoded))
    return [float(x) for x in bbox], area


def _resize_rgb(src_path: Path, dst_path: Path, image_size: int) -> None:
    arr = np.asarray(Image.open(src_path).convert("RGB"))
    if arr.shape[:2] != (image_size, image_size):
        arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    Image.fromarray(arr, mode="RGB").save(dst_path)


def _resize_depth(src_path: Path, dst_path: Path, image_size: int) -> np.ndarray:
    depth = np.load(src_path).astype(np.float32, copy=False)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.shape[:2] != (image_size, image_size):
        depth = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    np.save(dst_path, depth.astype(np.float32, copy=False))
    return depth


def _read_noise_mask(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        payload = np.load(path)
        if "arr_0" in payload:
            arr = payload["arr_0"]
        else:
            first_key = next(iter(payload.files))
            arr = payload[first_key]
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float32, copy=False)


def _write_noise_mask(src_path: Path, dst_path: Path, image_size: int) -> None:
    arr = _read_noise_mask(src_path)
    if arr.shape[:2] != (image_size, image_size):
        arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = src_path.suffix.lower()
    if suffix == ".npy":
        np.save(dst_path, arr.astype(np.float32, copy=False))
    elif suffix == ".npz":
        np.savez_compressed(dst_path, arr.astype(np.float32, copy=False))
    else:
        Image.fromarray(arr.astype(np.uint8)).save(dst_path)


def _optional_noise_mask_path(source_root: Path, split: str, stem: str) -> Path | None:
    mask_dir = source_root / "depth" / "depth_noise_mask" / split
    if not mask_dir.exists():
        return None
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".npy", ".npz"):
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _copy_and_patch_json(source_path: Path, target_path: Path, image_size: int) -> None:
    if not source_path.exists():
        return
    payload = _read_json(source_path)
    camera = payload.get("camera")
    if isinstance(camera, dict):
        if "width" in camera:
            camera["width"] = image_size
        if "height" in camera:
            camera["height"] = image_size
    if "build_dataset" in payload and isinstance(payload["build_dataset"], dict):
        payload["build_dataset"]["derived_image_size"] = image_size
    _write_json(target_path, payload)


@dataclass
class SplitSummary:
    split: str
    num_images: int
    num_annotations: int


def _build_split(
    source_root: Path,
    target_root: Path,
    split: str,
    image_size: int,
) -> SplitSummary:
    ann_path = source_root / "annotations" / f"instances_{split}.json"
    coco = _read_json(ann_path)
    images = list(coco.get("images", []))
    annotations = list(coco.get("annotations", []))
    by_image: Dict[int, List[dict[str, Any]]] = {}
    for ann in annotations:
        by_image.setdefault(int(ann["image_id"]), []).append(ann)

    target_images: List[dict[str, Any]] = []
    target_annotations: List[dict[str, Any]] = []
    split_manifest: List[dict[str, Any]] = []

    rgb_out_dir = target_root / "images" / split
    depth_out_dir = target_root / "depth" / "depth_npy" / split
    inst_out_dir = target_root / "cache" / "instance_map" / split
    xyz_out_dir = target_root / "cache" / "depth_xyz" / split
    rgb_out_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)
    inst_out_dir.mkdir(parents=True, exist_ok=True)
    xyz_out_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        stem = Path(file_name).stem
        src_rgb = source_root / "images" / split / file_name
        src_depth = source_root / "depth" / "depth_npy" / split / f"{stem}.npy"
        dst_rgb = rgb_out_dir / file_name
        dst_depth = depth_out_dir / f"{stem}.npy"

        _resize_rgb(src_rgb, dst_rgb, image_size=image_size)
        depth = _resize_depth(src_depth, dst_depth, image_size=image_size)
        depth_xyz = depth_to_xyz_like(depth)
        np.save(xyz_out_dir / f"{stem}.npy", depth_xyz.astype(np.float32, copy=False))

        instance_map = np.zeros((image_size, image_size), dtype=np.int32)
        next_instance_id = 1

        for ann in by_image.get(image_id, []):
            src_mask = _decode_segmentation(ann["segmentation"], int(image["height"]), int(image["width"]))
            resized_mask = cv2.resize(src_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            resized_mask = (resized_mask > 0).astype(np.uint8, copy=False)
            if resized_mask.max() <= 0:
                continue

            ann_out = dict(ann)
            ann_out["segmentation"] = _encode_mask(resized_mask)
            ann_out["bbox"], ann_out["area"] = _mask_bbox_and_area(resized_mask)
            target_annotations.append(ann_out)
            instance_map[resized_mask > 0] = next_instance_id
            next_instance_id += 1

        np.save(inst_out_dir / f"{stem}.npy", instance_map.astype(np.int32, copy=False))

        noise_path = _optional_noise_mask_path(source_root, split, stem)
        if noise_path is not None:
            dst_noise = target_root / "depth" / "depth_noise_mask" / split / noise_path.name
            _write_noise_mask(noise_path, dst_noise, image_size=image_size)

        image_out = dict(image)
        image_out["width"] = image_size
        image_out["height"] = image_size
        target_images.append(image_out)
        split_manifest.append(
            {
                "id": image_id,
                "file_name": file_name,
                "rgb_path": str(dst_rgb),
                "depth_path": str(dst_depth),
                "instance_map_path": str(inst_out_dir / f"{stem}.npy"),
                "depth_xyz_path": str(xyz_out_dir / f"{stem}.npy"),
            }
        )

    target_coco = dict(coco)
    target_coco["images"] = target_images
    target_coco["annotations"] = target_annotations

    _write_json(target_root / "annotations" / f"instances_{split}.json", target_coco)
    _write_json(target_root / "cache" / "splits" / f"{split}_images.json", split_manifest)
    return SplitSummary(split=split, num_images=len(target_images), num_annotations=len(target_annotations))


def build_multires_dataset(
    *,
    source_root: Path,
    target_root: Path,
    image_size: int,
    cache_root: Path,
    force: bool = False,
) -> Dict[str, Any]:
    source_root = source_root.resolve()
    target_root = target_root.resolve()
    _ensure_clean_dir(target_root, force=force)
    splits = _iter_splits(source_root)

    for name in ("dataset_info.json", "build_stats.json", "alignment_report.json", "mask_parity_report.json"):
        _copy_and_patch_json(source_root / name, target_root / name, image_size=image_size)

    summaries = [
        _build_split(source_root=source_root, target_root=target_root, split=split, image_size=image_size)
        for split in splits
    ]

    derived_manifest = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "image_size": int(image_size),
        "splits": [summary.split for summary in summaries],
        "summaries": [summary.__dict__ for summary in summaries],
    }
    _write_json(target_root / "cache" / "derived_dataset_manifest.json", derived_manifest)

    stats_payload = ensure_dataset_stats(dataset_root=target_root, cache_root=cache_root, force=force)
    preprocess_manifest = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "image_size": int(image_size),
        "instance_map_dir": str((target_root / "cache" / "instance_map").resolve()),
        "depth_xyz_dir": str((target_root / "cache" / "depth_xyz").resolve()),
        "split_manifest_dir": str((target_root / "cache" / "splits").resolve()),
        "stats_manifest_path": str(Path(stats_payload["manifest_path"]).resolve()),
        "rgb_stats_path": str(Path(stats_payload["rgb_stats_path"]).resolve()),
        "depth_stats_path": str(Path(stats_payload["depth_stats_path"]).resolve()),
    }
    _write_json(target_root / "cache" / "preprocess_manifest.json", preprocess_manifest)

    return {
        "derived_manifest_path": str((target_root / "cache" / "derived_dataset_manifest.json").resolve()),
        "preprocess_manifest_path": str((target_root / "cache" / "preprocess_manifest.json").resolve()),
        "stats_manifest_path": str(Path(stats_payload["manifest_path"]).resolve()),
        "target_root": str(target_root),
        "image_size": int(image_size),
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize an ECC derived dataset at a fixed square resolution.")
    parser.add_argument("--source-root", type=str, required=True)
    parser.add_argument("--target-root", type=str, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--cache-root", type=str, default=str(REPO_ROOT / "output" / "cache" / "dataset_stats"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = build_multires_dataset(
        source_root=Path(args.source_root),
        target_root=Path(args.target_root),
        image_size=int(args.image_size),
        cache_root=Path(args.cache_root),
        force=bool(args.force),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
