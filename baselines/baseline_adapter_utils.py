from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    return np.asarray(value)


def _as_1d_float32(values: Any | None) -> np.ndarray:
    if values is None:
        return np.zeros((0,), dtype=np.float32)
    arr = _to_numpy(values).astype(np.float32, copy=False)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _as_1d_int64(values: Any | None) -> np.ndarray:
    if values is None:
        return np.zeros((0,), dtype=np.int64)
    arr = _to_numpy(values).astype(np.int64, copy=False)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def decode_coco_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    from pycocotools import mask as mask_utils

    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            return np.zeros((height, width), dtype=np.uint8)
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
    else:
        raise TypeError(f"Unsupported COCO segmentation type: {type(segmentation)!r}")

    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


def binary_mask_to_rle(mask: Any) -> Dict[str, Any]:
    from pycocotools import mask as mask_utils

    arr = _to_numpy(mask)
    if arr.ndim == 3:
        arr = arr[0]
    arr = (arr > 0).astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(arr))
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("ascii")
    return {"size": list(rle["size"]), "counts": rle["counts"]}


def mask_to_bbox_xywh(mask: Any) -> List[float] | None:
    arr = _to_numpy(mask)
    if arr.ndim == 3:
        arr = arr[0]
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max() + 1)
    y1 = float(ys.max() + 1)
    return [x0, y0, x1 - x0, y1 - y0]


def annotations_to_instance_targets(
    annotations: Sequence[Mapping[str, Any]],
    height: int,
    width: int,
) -> Dict[str, Any]:
    instance_map = np.zeros((int(height), int(width)), dtype=np.int32)
    masks: List[np.ndarray] = []
    category_ids: List[int] = []
    annotation_ids: List[int] = []
    areas: List[float] = []
    bboxes: List[List[float] | None] = []

    for instance_id, annotation in enumerate(annotations, start=1):
        mask = decode_coco_segmentation(annotation.get("segmentation"), int(height), int(width))
        instance_map[mask > 0] = int(instance_id)
        masks.append(mask.astype(np.uint8, copy=False))
        category_ids.append(int(annotation.get("category_id", 0)))
        annotation_ids.append(int(annotation.get("id", instance_id)))
        areas.append(float(annotation.get("area", float(mask.sum()))))
        bboxes.append(mask_to_bbox_xywh(mask))

    return {
        "instance_map": instance_map,
        "masks": masks,
        "category_ids": np.asarray(category_ids, dtype=np.int64),
        "annotation_ids": np.asarray(annotation_ids, dtype=np.int64),
        "areas": np.asarray(areas, dtype=np.float32),
        "bboxes": bboxes,
    }


def resolve_instance_scores(
    scores: Any | None,
    masks: Sequence[Any] | np.ndarray | None = None,
    *,
    default_score: float = 1.0,
) -> np.ndarray:
    explicit = _as_1d_float32(scores)
    if explicit.size > 0:
        return explicit

    if masks is None:
        return np.zeros((0,), dtype=np.float32)

    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            mask_list = [masks]
        elif masks.ndim == 3:
            mask_list = [masks[i] for i in range(masks.shape[0])]
        else:
            mask_list = [masks]
    else:
        mask_list = list(masks)
    if len(mask_list) == 0:
        return np.zeros((0,), dtype=np.float32)

    fallback_scores = []
    for mask in mask_list:
        arr = _to_numpy(mask).astype(np.float32, copy=False)
        fallback_scores.append(float(arr.mean()) if arr.size > 0 else float(default_score))
    return np.asarray(fallback_scores, dtype=np.float32)


def _mask_to_binary(
    mask: Any,
    *,
    mask_threshold: float,
) -> np.ndarray:
    arr = _to_numpy(mask)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.dtype == np.uint8 and arr.max() <= 1:
        return (arr > 0).astype(np.uint8)
    if arr.dtype == np.uint8:
        return (arr > 127).astype(np.uint8)
    if arr.min() < 0.0 or arr.max() > 1.0:
        arr = 1.0 / (1.0 + np.exp(-arr))
    return (arr > float(mask_threshold)).astype(np.uint8)


def binary_masks_to_coco_rows(
    *,
    image_id: int,
    masks: Sequence[Any] | np.ndarray,
    scores: Any | None = None,
    category_ids: Sequence[int] | np.ndarray | None = None,
    score_threshold: float = 0.05,
    mask_threshold: float = 0.5,
    category_offset: int = 1,
) -> List[Dict[str, Any]]:
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            mask_list = [masks]
        elif masks.ndim == 3:
            mask_list = [masks[i] for i in range(masks.shape[0])]
        else:
            mask_list = [masks]
    else:
        mask_list = list(masks)
    score_arr = resolve_instance_scores(scores, mask_list)
    if category_ids is None:
        category_arr = np.zeros((len(mask_list),), dtype=np.int64)
    else:
        category_arr = _as_1d_int64(category_ids)

    rows: List[Dict[str, Any]] = []
    for idx, mask in enumerate(mask_list):
        score = float(score_arr[idx]) if idx < len(score_arr) else float(score_arr[-1]) if len(score_arr) else 0.0
        if score < float(score_threshold):
            continue
        binary_mask = _mask_to_binary(mask, mask_threshold=mask_threshold)
        bbox = mask_to_bbox_xywh(binary_mask)
        if bbox is None:
            continue
        contiguous_id = int(category_arr[idx]) if idx < len(category_arr) else 0
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": int(contiguous_id) + int(category_offset),
                "score": score,
                "bbox": bbox,
                "mask": binary_mask,
            }
        )
    return rows


def coco_rows_to_jsonable(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    jsonable: List[Dict[str, Any]] = []
    for row in rows:
        fixed = dict(row)
        mask = fixed.pop("mask", None)
        if mask is not None and "segmentation" not in fixed:
            fixed["segmentation"] = binary_mask_to_rle(mask)
        bbox = fixed.get("bbox")
        if bbox is not None:
            fixed["bbox"] = [float(v) for v in list(bbox)]
        if "score" in fixed:
            fixed["score"] = float(fixed["score"])
        if "image_id" in fixed:
            fixed["image_id"] = int(fixed["image_id"])
        if "category_id" in fixed:
            fixed["category_id"] = int(fixed["category_id"])
        jsonable.append(fixed)
    return jsonable


def write_baseline_run_artifacts(
    output_dir: str | Path,
    *,
    coco_rows: Sequence[Mapping[str, Any]] | None = None,
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    last_checkpoint: str | None = None,
    wall_time_sec: float | int | None = None,
    trainable_params: int | None = None,
) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Path] = {}

    if coco_rows is not None:
        results_path = out_dir / "coco_instances_results.json"
        results_path.write_text(
            json.dumps(coco_rows_to_jsonable(coco_rows), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        artifacts["coco_instances_results"] = results_path

    if metrics is not None:
        metrics_path = out_dir / "metrics.cocoeval.json"
        metrics_path.write_text(
            json.dumps(dict(metrics), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        artifacts["metrics_cocoeval"] = metrics_path

    metadata_path = out_dir / "metadata.json"
    existing_meta: Dict[str, Any] = {}
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_meta.update(loaded)
        except Exception:
            existing_meta = {}
    if metadata is not None:
        existing_meta.update(dict(metadata))
        metadata_path.write_text(
            json.dumps(existing_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif metadata_path.exists():
        artifacts["metadata"] = metadata_path
    if metadata is not None:
        artifacts["metadata"] = metadata_path

    if last_checkpoint is not None:
        last_checkpoint_path = out_dir / "last_checkpoint"
        last_checkpoint_path.write_text(f"{last_checkpoint}\n", encoding="utf-8")
        artifacts["last_checkpoint"] = last_checkpoint_path

    if wall_time_sec is not None:
        wall_time_path = out_dir / "wall_time_sec.txt"
        wall_time_path.write_text(f"{wall_time_sec}\n", encoding="utf-8")
        artifacts["wall_time_sec"] = wall_time_path

    if trainable_params is not None:
        params_path = out_dir / "params_trainable.txt"
        params_path.write_text(f"{int(trainable_params)}\n", encoding="utf-8")
        artifacts["params_trainable"] = params_path

    return artifacts
