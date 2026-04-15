from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

try:
    from .baseline_adapter_utils import binary_masks_to_coco_rows
    from .ecc_data_utils import load_ecc_coco_rgb_image, load_ecc_coco_rgb_records
except ImportError:  # pragma: no cover - file execution fallback
    from baseline_adapter_utils import binary_masks_to_coco_rows
    from ecc_data_utils import load_ecc_coco_rgb_image, load_ecc_coco_rgb_records


SUPPORTED_IMAGE_SIZES = (512, 1024)


def _resize_instance_map(instance_map: np.ndarray, image_size: int) -> np.ndarray:
    if instance_map.shape[:2] == (int(image_size), int(image_size)):
        return instance_map.astype(np.int32, copy=False)
    resized = cv2.resize(
        instance_map.astype(np.int32, copy=False),
        (int(image_size), int(image_size)),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(np.int32, copy=False)


def load_stardist_ecc_split(
    dataset_root: str | Path,
    split: str,
    image_size: int,
    *,
    max_images: int | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
    records = load_ecc_coco_rgb_records(dataset_root, split)
    if max_images is not None and int(max_images) > 0:
        records = records[: int(max_images)]

    images: List[np.ndarray] = []
    label_maps: List[np.ndarray] = []
    selected_records: List[Dict[str, Any]] = []
    for record in records:
        image = load_ecc_coco_rgb_image(record["image_path"], image_size=int(image_size))
        instance_map = _resize_instance_map(
            np.asarray(record["annotation_targets"]["instance_map"], dtype=np.int32),
            int(image_size),
        )
        images.append(image.astype(np.uint8, copy=False))
        label_maps.append(instance_map)
        selected_records.append(record)
    return images, label_maps, selected_records


def stardist_prediction_to_coco_rows(
    *,
    image_id: int,
    labels: np.ndarray,
    details: Mapping[str, Any] | Dict[str, Any] | None,
    score_threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label image, got shape={labels.shape!r}")

    object_ids = [int(label_id) for label_id in np.unique(labels).tolist() if int(label_id) > 0]
    masks = [(labels == label_id).astype(np.uint8, copy=False) for label_id in object_ids]
    scores = None if details is None else details.get("prob")
    category_ids = np.zeros((len(masks),), dtype=np.int64)
    return binary_masks_to_coco_rows(
        image_id=int(image_id),
        masks=masks,
        scores=scores,
        category_ids=category_ids,
        score_threshold=float(score_threshold),
        mask_threshold=0.5,
        category_offset=1,
    )
