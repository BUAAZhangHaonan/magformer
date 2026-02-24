from __future__ import annotations

from typing import Optional, Tuple


def _normalize_register(register: str) -> str:
    r = (register or "").strip().lower()
    if r in {"0831", "0831_1k", "ecc0831", "ecc0831_1k"}:
        return "0831"
    if r in {"0909", "0909_512", "ecc0909", "ecc0909_512"}:
        return "0909"
    raise ValueError(f"Unsupported --register value: {register!r} (expected: 0831|0909)")


def register_ecc_coco(register: str, dataset_root: Optional[str] = None) -> Tuple[str, str]:
    """
    Register ECC datasets for Detectron2 baselines (RGB-only).

    Args:
        register: "0831" or "0909"
        dataset_root: path to dataset root (optional; falls back to per-dataset defaults/env).

    Returns:
        (train_name, val_name)
    """
    r = _normalize_register(register)
    if r == "0831":
        from baselines.register_0831_1k_coco import (
            DATASET_NAME_TRAIN,
            DATASET_NAME_VAL,
            register_0831_1k_coco,
        )

        register_0831_1k_coco(dataset_root)
        return DATASET_NAME_TRAIN, DATASET_NAME_VAL

    from baselines.register_0909_512_coco import (
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
    r = _normalize_register(register)
    if r == "0831":
        from baselines.register_0831_1k_coco_rgbd import (
            DATASET_NAME_TRAIN,
            DATASET_NAME_VAL,
            register_0831_1k_coco_rgbd,
        )

        register_0831_1k_coco_rgbd(dataset_root)
        return DATASET_NAME_TRAIN, DATASET_NAME_VAL

    from baselines.register_0909_512_coco_rgbd import (
        DATASET_NAME_TRAIN,
        DATASET_NAME_VAL,
        register_0909_512_coco_rgbd,
    )

    register_0909_512_coco_rgbd(dataset_root)
    return DATASET_NAME_TRAIN, DATASET_NAME_VAL

