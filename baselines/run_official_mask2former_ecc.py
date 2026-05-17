#!/usr/bin/env python3
"""
Run official facebookresearch/Mask2Former training with ECC datasets (0831 / 0909).

Why this wrapper exists:
- Detectron2 requires datasets to be registered in-process.
- We keep official baseline repos as submodules (no patching inside them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Prefer a compiled Detectron2 from the active environment before adding the
# local baselines directory, which also contains an unbuilt Detectron2 checkout.
try:
    import detectron2  # noqa: F401
except Exception:
    pass

# Ensure sibling baseline utilities are importable when running as a file.
BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from ecc_datasets import register_ecc_coco
from ecc_datasets import dataset_prefix


def _split_args(argv: List[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def _resolve_dataset_path(dataset_root: Optional[str], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if dataset_root is None:
        return path.resolve()
    return (Path(dataset_root) / path).resolve()


def _rle_to_polygons(segmentation: dict) -> list[list[float]]:
    from pycocotools import mask as mask_utils
    import cv2
    import numpy as np

    height, width = segmentation["size"]
    rle = segmentation
    if isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    mask = np.ascontiguousarray(decoded.astype("uint8"))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    polygons: list[list[float]] = []
    for contour in contours:
        polygon = contour.reshape(-1, 2).astype(float).reshape(-1).tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)

    if not polygons and mask.any():
        x, y, width, height = cv2.boundingRect(mask)
        polygons.append([
            float(x),
            float(y),
            float(x + width),
            float(y),
            float(x + width),
            float(y + height),
            float(x),
            float(y + height),
        ])
    return polygons


def _normalize_rle_segmentations(payload: dict) -> tuple[list[dict], bool]:
    normalized_annotations = []
    changed = False
    for annotation in payload.get("annotations", []):
        normalized_annotation = dict(annotation)
        segmentation = normalized_annotation.get("segmentation")
        if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
            normalized_annotation["segmentation"] = _rle_to_polygons(segmentation)
            changed = True
        normalized_annotations.append(normalized_annotation)
    return normalized_annotations, changed


def _normalize_coco_metadata(ann_file: Path, normalized_ann_dir: str) -> Path:
    if not ann_file.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

    with ann_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"COCO annotation must be a JSON object: {ann_file}")

    normalized_annotations, converted_rle = _normalize_rle_segmentations(payload)
    has_info = isinstance(payload.get("info"), dict)
    has_licenses = isinstance(payload.get("licenses"), list)
    if has_info and has_licenses and not converted_rle:
        return ann_file

    normalized = dict(payload)
    normalized["annotations"] = normalized_annotations
    normalized.setdefault(
        "info",
        {
            "description": "Normalized temporary COCO annotation for official Mask2Former wrapper",
            "source_annotation": str(ann_file),
        },
    )
    normalized.setdefault("licenses", [])

    digest = hashlib.sha1(str(ann_file).encode("utf-8")).hexdigest()[:10]
    out_dir = Path(normalized_ann_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{ann_file.stem}.{digest}.normalized.json"
    out_file.write_text(json.dumps(normalized, ensure_ascii=False), encoding="utf-8")
    print(f"[official-mask2former] normalized COCO metadata: {ann_file} -> {out_file}", flush=True)
    return out_file


def _register_coco_split(
    *,
    name: str,
    ann_file: Path,
    image_root: Path,
    normalized_ann_dir: str,
) -> None:
    if not image_root.exists():
        raise FileNotFoundError(f"COCO image directory not found: {image_root}")

    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets.coco import load_coco_json

    normalized_ann = _normalize_coco_metadata(ann_file, normalized_ann_dir)

    if name not in DatasetCatalog:
        DatasetCatalog.register(
            name,
            lambda ann_file=normalized_ann, image_root=image_root, name=name: load_coco_json(
                str(ann_file), str(image_root), dataset_name=name
            ),
        )
    MetadataCatalog.get(name).set(
        thing_classes=["component"],
        evaluator_type="coco",
        image_root=str(image_root),
        json_file=str(normalized_ann),
    )


def _load_coco_sqlite_records(cache_path: Path, image_root: Path) -> list[dict]:
    if not cache_path.exists():
        raise FileNotFoundError(f"SQLite cache not found: {cache_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"COCO image directory not found: {image_root}")

    from detectron2.structures import BoxMode

    category_id_to_contiguous_id = {1: 0}
    annotations_by_image_id: dict[int, list[dict]] = {}

    with sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True) as conn:
        for image_id, annotation_json in conn.execute(
            "SELECT image_id, json FROM annotations ORDER BY image_id, ordinal"
        ):
            annotation = json.loads(annotation_json)
            category_id = int(annotation["category_id"])
            if category_id not in category_id_to_contiguous_id:
                raise ValueError(f"Unsupported category_id {category_id} in SQLite cache: {cache_path}")
            annotations_by_image_id.setdefault(int(image_id), []).append(
                {
                    "bbox": annotation["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": annotation.get("segmentation", []),
                    "category_id": category_id_to_contiguous_id[category_id],
                    "iscrowd": int(annotation.get("iscrowd", 0)),
                }
            )

        records = []
        for image_id, file_name, image_json in conn.execute("SELECT id, file_name, json FROM images ORDER BY id"):
            image = json.loads(image_json)
            image_id = int(image_id)
            records.append(
                {
                    "file_name": str((image_root / file_name).resolve()),
                    "height": int(image["height"]),
                    "width": int(image["width"]),
                    "image_id": image_id,
                    "annotations": annotations_by_image_id.get(image_id, []),
                }
            )

    return records


def _register_coco_sqlite_train_split(
    *,
    name: str,
    cache_path: Path,
    image_root: Path,
) -> None:
    if not cache_path.exists():
        raise FileNotFoundError(f"SQLite cache not found: {cache_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"COCO image directory not found: {image_root}")

    from detectron2.data import DatasetCatalog, MetadataCatalog

    if name not in DatasetCatalog:
        DatasetCatalog.register(
            name,
            lambda cache_path=cache_path, image_root=image_root: _load_coco_sqlite_records(cache_path, image_root),
        )
    metadata = MetadataCatalog.get(name)
    metadata.set(
        thing_classes=["component"],
        thing_dataset_id_to_contiguous_id={1: 0},
        evaluator_type="coco",
        image_root=str(image_root),
    )


def register_official_mask2former_datasets(
    *,
    register: str,
    dataset_root: Optional[str],
    train_ann: Optional[str] = None,
    train_cache: Optional[str] = None,
    val_ann: Optional[str] = None,
    train_image_dir: Optional[str] = None,
    val_image_dir: Optional[str] = None,
    train_split: str = "train",
    val_split: str = "val",
    normalized_ann_dir: str = "output/diagnostics/official_mask2former_coco",
) -> Tuple[str, str]:
    """
    Register the RGB COCO datasets used by the official Mask2Former wrapper.

    Explicit annotation/image-root arguments are used for pseudo-real splits
    such as target_labeled and target_unlabeled. Missing annotation files fail
    immediately; source COCO JSON files are never modified in place.
    """
    if train_ann is None and train_cache is None and val_ann is None and train_image_dir is None and val_image_dir is None:
        return register_ecc_coco(register, dataset_root)

    if dataset_root is None:
        raise ValueError("--dataset-root is required when using explicit official Mask2Former split arguments")

    prefix = dataset_prefix(register, dataset_root)
    train_name = f"{prefix}_{train_split}"
    val_name = f"{prefix}_{val_split}"

    train_ann_path = None
    train_cache_path = None
    if train_cache is None:
        train_ann_path = _resolve_dataset_path(dataset_root, train_ann or f"annotations/instances_{train_split}.json")
    else:
        train_cache_path = _resolve_dataset_path(dataset_root, train_cache)
    val_ann_path = _resolve_dataset_path(dataset_root, val_ann or f"annotations/instances_{val_split}.json")
    train_image_root = _resolve_dataset_path(dataset_root, train_image_dir or f"images/{train_split}")
    val_image_root = _resolve_dataset_path(dataset_root, val_image_dir or f"images/{val_split}")

    if train_cache_path is None:
        _register_coco_split(
            name=train_name,
            ann_file=train_ann_path,
            image_root=train_image_root,
            normalized_ann_dir=normalized_ann_dir,
        )
    else:
        _register_coco_sqlite_train_split(
            name=train_name,
            cache_path=train_cache_path,
            image_root=train_image_root,
        )
    _register_coco_split(
        name=val_name,
        ann_file=val_ann_path,
        image_root=val_image_root,
        normalized_ann_dir=normalized_ann_dir,
    )
    print(f"[official-mask2former] registered train dataset: {train_name}", flush=True)
    print(f"[official-mask2former] registered val/test dataset: {val_name}", flush=True)
    return train_name, val_name


def main() -> None:
    wrapper_argv, passthrough = _split_args(sys.argv[1:])

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--register",
        type=str,
        default="0831",
        help="ECC dataset id: 0831 | 0909",
    )
    ap.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to magformer_datasets/<dataset> root (default: env + workspace-relative).",
    )
    ap.add_argument(
        "--mask2former-root",
        type=str,
        default="baselines/Mask2Former",
        help="Path to official Mask2Former repo checkout.",
    )
    ap.add_argument("--train-ann", type=str, default=None, help="Train COCO annotation file, absolute or relative to --dataset-root.")
    ap.add_argument("--train-cache", type=str, default=None, help="Train COCO SQLite cache, absolute or relative to --dataset-root.")
    ap.add_argument("--val-ann", type=str, default=None, help="Val/test COCO annotation file, absolute or relative to --dataset-root.")
    ap.add_argument("--train-image-dir", type=str, default=None, help="Train image directory, absolute or relative to --dataset-root.")
    ap.add_argument("--val-image-dir", type=str, default=None, help="Val/test image directory, absolute or relative to --dataset-root.")
    ap.add_argument("--train-split", type=str, default="train", help="Train split suffix used in the registered dataset name.")
    ap.add_argument("--val-split", type=str, default="val", help="Val/test split suffix used in the registered dataset name.")
    ap.add_argument(
        "--normalized-ann-dir",
        type=str,
        default="output/diagnostics/official_mask2former_coco",
        help="Directory for temporary COCO JSON files with required info/licenses metadata.",
    )
    args = ap.parse_args(wrapper_argv)

    register_official_mask2former_datasets(
        register=args.register,
        dataset_root=args.dataset_root,
        train_ann=args.train_ann,
        train_cache=args.train_cache,
        val_ann=args.val_ann,
        train_image_dir=args.train_image_dir,
        val_image_dir=args.val_image_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        normalized_ann_dir=args.normalized_ann_dir,
    )

    repo_root = Path(args.mask2former_root).resolve()
    train_py = repo_root / "train_net.py"
    if not train_py.exists():
        raise FileNotFoundError(f"train_net.py not found: {train_py}")

    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    sys.argv = [str(train_py)] + passthrough
    runpy.run_path(str(train_py), run_name="__main__")


if __name__ == "__main__":
    main()
