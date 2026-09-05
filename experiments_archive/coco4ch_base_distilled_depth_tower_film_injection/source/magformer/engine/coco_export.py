# -*- coding: utf-8 -*-
"""Shared COCO export helpers used by both online and offline evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from pycocotools import mask as coco_mask


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    return np.asarray(x)


def _mask_to_binary(
    mask: Any,
    mask_threshold: float = 0.5,
    allow_empty_fallback: bool = False,
    empty_fallback_ratio: float = 0.01,
) -> np.ndarray:
    arr = _to_numpy(mask)
    if not np.isfinite(arr).all():
        raise ValueError("Mask contains NaN or Inf values")
    if arr.ndim == 3:
        arr = arr[0]
    if arr.dtype == np.uint8:
        if arr.max() <= 1:
            return (arr > 0).astype(np.uint8)
        return (arr > 127).astype(np.uint8)
    if arr.min() < 0.0 or arr.max() > 1.0:
        probs = 1.0 / (1.0 + np.exp(-arr))
    else:
        probs = arr
    binary = (probs > float(mask_threshold)).astype(np.uint8)
    if binary.sum() > 0 or not allow_empty_fallback:
        return binary

    flat = probs.reshape(-1)
    ratio = max(0.0, min(1.0, float(empty_fallback_ratio)))
    if ratio == 0.0:
        return binary
    topk = max(1, int(round(ratio * flat.size)))
    top_idx = np.argpartition(flat, -topk)[-topk:]
    fallback = np.zeros_like(flat, dtype=np.uint8)
    fallback[top_idx] = 1
    return fallback.reshape(probs.shape)


def _encode_mask_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Encode binary mask to COCO RLE format (~100 bytes vs ~256KB raw)."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    rle = coco_mask.encode(np.asfortranarray(mask))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return {"size": rle["size"], "counts": rle["counts"]}


def _bbox_xyxy_from_binary_mask(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return [x1, y1, x2, y2]


def predictions_to_coco_instances(
    predictions: Iterable[Dict[str, Any]],
    image_ids: Iterable[int],
    score_threshold: float = 0.0,
    mask_threshold: float = 0.5,
    category_offset: int = 1,
    category_ids: Optional[Iterable[int]] = None,
    allow_empty_fallback: bool = False,
    empty_fallback_ratio: float = 0.01,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pred_list = list(predictions)
    if image_ids is None:
        raise ValueError("image_ids are required for COCO export")
    img_ids = [int(image_id) for image_id in image_ids]
    if len(img_ids) != len(pred_list):
        raise ValueError(
            "image_ids length must match predictions length: "
            f"image_ids={len(img_ids)}, predictions={len(pred_list)}"
        )
    category_id_list = list(category_ids) if category_ids is not None else None

    for batch_idx, pred in enumerate(pred_list):
        image_id = img_ids[batch_idx]

        missing_keys = [key for key in ("scores", "masks") if key not in pred]
        if "category_ids" not in pred and "labels" not in pred:
            missing_keys.append("category_ids")
        if missing_keys:
            raise KeyError(
                f"Prediction for image_id={image_id} is missing required keys: "
                f"{missing_keys}"
            )

        scores = pred["scores"]
        category_ids = pred.get("category_ids", pred.get("labels"))
        masks = pred["masks"]

        scores_arr = _to_numpy(scores) if len(
            scores) > 0 else np.zeros((0,), dtype=np.float32)
        cat_arr = _to_numpy(category_ids) if len(
            category_ids) > 0 else np.zeros((len(scores_arr),), dtype=np.int64)
        masks_arr = _to_numpy(masks) if len(
            masks) > 0 else np.zeros((0,), dtype=np.float32)

        if masks_arr.ndim == 2:
            masks_arr = masks_arr[None, ...]
        if len(scores_arr) != len(cat_arr) or len(scores_arr) != len(masks_arr):
            raise ValueError(
                f"Prediction length mismatch for image_id={image_id}: "
                f"scores={len(scores_arr)}, category_ids={len(cat_arr)}, "
                f"masks={len(masks_arr)}"
            )

        for i in range(len(scores_arr)):
            score = float(scores_arr[i])
            if score < float(score_threshold):
                continue
            if i >= len(masks_arr):
                continue

            binary_mask = _mask_to_binary(
                masks_arr[i],
                mask_threshold=mask_threshold,
                allow_empty_fallback=allow_empty_fallback,
                empty_fallback_ratio=empty_fallback_ratio,
            )
            bbox = _bbox_xyxy_from_binary_mask(binary_mask)
            if bbox is None:
                continue

            # Encode to RLE immediately to save memory (~100B vs ~256KB)
            rle_mask = _encode_mask_rle(binary_mask)
            del binary_mask

            if i < len(cat_arr):
                contiguous_id = int(cat_arr[i])
                if category_id_list is not None and 0 <= contiguous_id < len(category_id_list):
                    category_id = int(category_id_list[contiguous_id])
                else:
                    category_id = contiguous_id + int(category_offset)
            else:
                category_id = int(category_id_list[0]) if category_id_list else int(category_offset)

            rows.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "score": score,
                    "mask": rle_mask,
                    "bbox": bbox,
                }
            )

    return rows


def outputs_to_coco_instances(
    outputs: Dict[str, Any],
    image_ids: Iterable[int],
    score_threshold: float = 0.0,
    mask_threshold: float = 0.5,
    category_offset: int = 1,
    category_ids: Optional[Iterable[int]] = None,
    allow_empty_fallback: bool = False,
    empty_fallback_ratio: float = 0.01,
) -> List[Dict[str, Any]]:
    predictions = outputs.get("predictions", None)
    if predictions is None:
        raise KeyError("outputs must contain a 'predictions' key")
    return predictions_to_coco_instances(
        predictions=predictions,
        image_ids=image_ids,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        category_offset=category_offset,
        category_ids=category_ids,
        allow_empty_fallback=allow_empty_fallback,
        empty_fallback_ratio=empty_fallback_ratio,
    )
