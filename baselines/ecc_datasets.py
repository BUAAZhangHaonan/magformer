from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple


def _normalize_register(register: str) -> str:
    r = (register or "").strip().lower()
    if r in {"0831", "0831_1k", "ecc0831", "ecc0831_1k"}:
        return "0831"
    if r in {"0909", "0909_512", "ecc0909", "ecc0909_512"}:
        return "0909"
    raise ValueError(
        f"Unsupported --register value: {register!r} (expected: 0831|0909)")


def _slugify_dataset_id(register: str, dataset_root: Optional[str]) -> str:
    source = Path(dataset_root).name if dataset_root else register
    slug = re.sub(r"[^a-z0-9]+", "_", str(source).strip().lower()).strip("_")
    if not slug:
        raise ValueError(f"Could not derive dataset slug from register={register!r}, dataset_root={dataset_root!r}")
    if slug.startswith("ecc"):
        return slug
    return f"ecc{slug}"


def dataset_prefix(register: str, dataset_root: Optional[str] = None) -> str:
    try:
        normalized = _normalize_register(register)
    except ValueError:
        return _slugify_dataset_id(register, dataset_root)
    if normalized == "0831":
        return "ecc0831_1k"
    return "ecc0909_512"


def dataset_name_pair_coco(register: str, dataset_root: Optional[str] = None) -> Tuple[str, str]:
    prefix = dataset_prefix(register, dataset_root)
    return f"{prefix}_train", f"{prefix}_val"


def dataset_name_pair_coco_rgbd(register: str, dataset_root: Optional[str] = None) -> Tuple[str, str]:
    prefix = dataset_prefix(register, dataset_root)
    return f"{prefix}_rgbd_train", f"{prefix}_rgbd_val"


def _register_custom_coco(dataset_root: str) -> Tuple[str, str]:
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import DatasetCatalog, MetadataCatalog

    root = Path(dataset_root).resolve()
    train_name, val_name = dataset_name_pair_coco(root.name, str(root))

    for name, split in ((train_name, "train"), (val_name, "val")):
        img_dir = root / "images" / split
        ann_file = root / "annotations" / f"instances_{split}.json"
        if name not in DatasetCatalog:
            register_coco_instances(name, {}, str(ann_file), str(img_dir))
        MetadataCatalog.get(name).set(thing_classes=["component"])

    return train_name, val_name


def _resolve_depth_dir(root: Path, split: str) -> Path:
    candidates = [
        root / "depth" / "depth_npy" / split,
        root / "depth" / split,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _register_custom_coco_rgbd(dataset_root: str) -> Tuple[str, str]:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets.coco import load_coco_json

    root = Path(dataset_root).resolve()
    train_name, val_name = dataset_name_pair_coco_rgbd(root.name, str(root))

    for name, split in ((train_name, "train"), (val_name, "val")):
        img_dir = root / "images" / split
        depth_dir = _resolve_depth_dir(root, split)
        ann_file = root / "annotations" / f"instances_{split}.json"

        def _loader(ann_file=ann_file, img_dir=img_dir, depth_dir=depth_dir, name=name):
            ds = load_coco_json(str(ann_file), str(img_dir), dataset_name=name)
            for record in ds:
                stem = Path(record["file_name"]).stem
                depth_path = depth_dir / f"{stem}.npy"
                if not depth_path.exists():
                    raise FileNotFoundError(f"Missing depth file for {record['file_name']}: {depth_path}")
                record["depth_file_name"] = str(depth_path)
            return ds

        if name not in DatasetCatalog:
            DatasetCatalog.register(name, _loader)
        MetadataCatalog.get(name).set(
            thing_classes=["component"],
            evaluator_type="coco",
            image_root=str(img_dir),
            json_file=str(ann_file),
        )

    return train_name, val_name


def normalize_register(register: str) -> str:
    try:
        return _normalize_register(register)
    except ValueError:
        slug = re.sub(r"[^a-z0-9]+", "_", str(register).strip().lower()).strip("_")
        if slug.startswith("ecc"):
            slug = slug[3:]
        if not slug:
            raise ValueError(f"Unsupported --register value: {register!r}")
        return slug


def register_ecc_coco(register: str, dataset_root: Optional[str] = None) -> Tuple[str, str]:
    """
    Register ECC datasets for Detectron2 baselines (RGB-only).

    Args:
        register: "0831" or "0909"
        dataset_root: path to dataset root (optional; falls back to per-dataset defaults/env).

    Returns:
        (train_name, val_name)
    """
    try:
        r = _normalize_register(register)
    except ValueError:
        if dataset_root is None:
            raise
        return _register_custom_coco(dataset_root)
    if r == "0831":
        try:
            from .register_0831_1k_coco import (  # type: ignore[import-not-found]
                DATASET_NAME_TRAIN,
                DATASET_NAME_VAL,
                register_0831_1k_coco,
            )
        except ImportError:
            from register_0831_1k_coco import (
                DATASET_NAME_TRAIN,
                DATASET_NAME_VAL,
                register_0831_1k_coco,
            )

        register_0831_1k_coco(dataset_root)
        return DATASET_NAME_TRAIN, DATASET_NAME_VAL

    try:
        from .register_0909_512_coco import (  # type: ignore[import-not-found]
            DATASET_NAME_TRAIN,
            DATASET_NAME_VAL,
            register_0909_512_coco,
        )
    except ImportError:
        from register_0909_512_coco import (
            DATASET_NAME_TRAIN,
            DATASET_NAME_VAL,
            register_0909_512_coco,
        )

    register_0909_512_coco(dataset_root)
    return DATASET_NAME_TRAIN, DATASET_NAME_VAL


def register_ecc_coco_rgbd(register: str, dataset_root: Optional[str] = None) -> Tuple[str, str]:
    """
    Register ECC datasets for Detectron2 RGBD baselines (adds `depth_file_name`).

    Args:
        register: "0831" or "0909"
        dataset_root: path to dataset root (optional; falls back to per-dataset defaults/env).

    Returns:
        (train_name, val_name)
    """
    try:
        r = _normalize_register(register)
    except ValueError:
        if dataset_root is None:
            raise
        return _register_custom_coco_rgbd(dataset_root)
    if r == "0831":
        try:
            from .register_0831_1k_coco_rgbd import (  # type: ignore[import-not-found]
                DATASET_NAME_TRAIN,
                DATASET_NAME_VAL,
                register_0831_1k_coco_rgbd,
            )
        except ImportError:
            from register_0831_1k_coco_rgbd import (
                DATASET_NAME_TRAIN,
                DATASET_NAME_VAL,
                register_0831_1k_coco_rgbd,
            )

        register_0831_1k_coco_rgbd(dataset_root)
        return DATASET_NAME_TRAIN, DATASET_NAME_VAL

    try:
        from .register_0909_512_coco_rgbd import (  # type: ignore[import-not-found]
            DATASET_NAME_TRAIN,
            DATASET_NAME_VAL,
            register_0909_512_coco_rgbd,
        )
    except ImportError:
        from register_0909_512_coco_rgbd import (
            DATASET_NAME_TRAIN,
            DATASET_NAME_VAL,
            register_0909_512_coco_rgbd,
        )

    register_0909_512_coco_rgbd(dataset_root)
    return DATASET_NAME_TRAIN, DATASET_NAME_VAL
