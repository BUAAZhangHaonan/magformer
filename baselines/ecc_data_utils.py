from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .baseline_adapter_utils import annotations_to_instance_targets


def _resolve_ecc_image_dir(dataset_root: Path, split: str) -> Path:
    return dataset_root / "images" / split


def _resolve_ecc_annotation_path(dataset_root: Path, split: str) -> Path:
    return dataset_root / "annotations" / f"instances_{split}.json"


def load_ecc_coco_rgb_records(dataset_root: str | Path, split: str) -> List[Dict[str, Any]]:
    root = Path(dataset_root)
    ann_path = _resolve_ecc_annotation_path(root, split)
    img_dir = _resolve_ecc_image_dir(root, split)
    payload = json.loads(ann_path.read_text(encoding="utf-8"))

    records: List[Dict[str, Any]] = []
    for image_info in payload.get("images", []):
        image_path = img_dir / image_info["file_name"]
        annotations = [
            ann
            for ann in payload.get("annotations", [])
            if int(ann.get("image_id", -1)) == int(image_info["id"])
        ]
        records.append(
            {
                "image_id": int(image_info["id"]),
                "file_name": image_info["file_name"],
                "image_path": str(image_path),
                "height": int(image_info["height"]),
                "width": int(image_info["width"]),
                "annotations": annotations,
                "annotation_targets": annotations_to_instance_targets(
                    annotations,
                    height=int(image_info["height"]),
                    width=int(image_info["width"]),
                ),
            }
        )
    return records


def load_ecc_coco_rgb_image(image_path: str | Path, image_size: int | None = None) -> np.ndarray:
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        image = cv2.resize(image, (int(image_size), int(image_size)), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.uint8, copy=False)

