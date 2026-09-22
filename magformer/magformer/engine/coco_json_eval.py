"""Canonical COCO JSON validation and official metric computation.

The file-boundary schema in this module is the standard COCO result schema:
``bbox`` is always ``[x, y, width, height]`` and ``segmentation`` is a
compressed COCO RLE.  Internal model formats such as XYXY boxes must be
converted before calling this module.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STANDARD_IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
STANDARD_RECALL_THRESHOLDS = tuple(float(value) for value in np.linspace(0.0, 1.0, 101))
STANDARD_AREA_RANGES = (
    (0.0, 1.0e10),
    (0.0, float(32**2)),
    (float(32**2), float(96**2)),
    (float(96**2), 1.0e10),
)
STANDARD_AREA_LABELS = ("all", "small", "medium", "large")
SUPPORTED_IOU_TYPES = ("bbox", "segm")


@dataclass(frozen=True)
class COCOEvalContract:
    """Immutable parameters shared by every COCOeval instance in one run."""

    image_ids: tuple[int, ...]
    category_ids: tuple[int, ...]
    iou_thresholds: tuple[float, ...]
    recall_thresholds: tuple[float, ...]
    max_dets: tuple[int, int, int]
    area_ranges: tuple[tuple[float, float], ...]
    area_labels: tuple[str, ...]
    use_categories: bool = True

    def apply(self, coco_eval: COCOeval) -> None:
        """Apply an independent copy of this contract to a COCOeval object."""

        coco_eval.params.imgIds = list(self.image_ids)
        coco_eval.params.catIds = list(self.category_ids)
        coco_eval.params.iouThrs = np.asarray(self.iou_thresholds, dtype=np.float64)
        coco_eval.params.recThrs = np.asarray(self.recall_thresholds, dtype=np.float64)
        coco_eval.params.maxDets = list(self.max_dets)
        coco_eval.params.areaRng = [list(area_range) for area_range in self.area_ranges]
        coco_eval.params.areaRngLbl = list(self.area_labels)
        coco_eval.params.useCats = int(self.use_categories)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_ids": list(self.image_ids),
            "category_ids": list(self.category_ids),
            "iou_thresholds": list(self.iou_thresholds),
            "recall_thresholds": list(self.recall_thresholds),
            "max_dets": list(self.max_dets),
            "area_ranges": [list(area_range) for area_range in self.area_ranges],
            "area_labels": list(self.area_labels),
            "use_categories": self.use_categories,
        }


def _require_unique_ints(values: Iterable[Any], *, field: str) -> tuple[int, ...]:
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{field} must contain integers, got {value!r}")
        normalized.append(int(value))
    duplicates = sorted(value for value, count in Counter(normalized).items() if count > 1)
    if duplicates:
        raise ValueError(f"{field} contains duplicates: {duplicates}")
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return tuple(sorted(normalized))


def _validate_iou_thresholds(values: Iterable[Any]) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in values)
    if not thresholds:
        raise ValueError("iou_thresholds must not be empty")
    array = np.asarray(thresholds, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("iou_thresholds must contain only finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("iou_thresholds must be within [0, 1]")
    if np.any(np.diff(array) <= 0.0):
        raise ValueError("iou_thresholds must be strictly increasing")
    return thresholds


def build_coco_eval_contract(
    coco_gt: COCO,
    *,
    image_ids: Iterable[int] | None = None,
    max_dets: int = 100,
    iou_thresholds: Iterable[float] = STANDARD_IOU_THRESHOLDS,
) -> COCOEvalContract:
    """Build the exact COCO parameter contract for an evaluation run."""

    if isinstance(max_dets, bool) or not isinstance(max_dets, (int, np.integer)):
        raise TypeError(f"max_dets must be an integer greater than 10, got {max_dets!r}")
    max_dets = int(max_dets)
    if max_dets <= 10:
        raise ValueError(f"max_dets must be greater than 10, got {max_dets}")

    gt_image_ids = _require_unique_ints(coco_gt.getImgIds(), field="GT image IDs")
    selected_image_ids = (
        gt_image_ids
        if image_ids is None
        else _require_unique_ints(image_ids, field="evaluated image IDs")
    )
    unknown_image_ids = sorted(set(selected_image_ids) - set(gt_image_ids))
    if unknown_image_ids:
        raise ValueError(f"evaluated image IDs are absent from COCO GT: {unknown_image_ids}")

    category_ids = _require_unique_ints(coco_gt.getCatIds(), field="GT category IDs")
    return COCOEvalContract(
        image_ids=selected_image_ids,
        category_ids=category_ids,
        iou_thresholds=_validate_iou_thresholds(iou_thresholds),
        recall_thresholds=STANDARD_RECALL_THRESHOLDS,
        max_dets=(1, 10, max_dets),
        area_ranges=STANDARD_AREA_RANGES,
        area_labels=STANDARD_AREA_LABELS,
        use_categories=True,
    )


def _require_row_int(row: Mapping[str, Any], field: str, *, row_index: int) -> int:
    if field not in row:
        raise ValueError(f"prediction row {row_index} is missing {field!r}")
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"prediction row {row_index} field {field!r} must be an integer")
    return int(value)


def _canonical_bbox(value: Any, *, row_index: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"prediction row {row_index} bbox must be COCO XYWH with four values")
    bbox = [float(component) for component in value]
    if not np.isfinite(np.asarray(bbox, dtype=np.float64)).all():
        raise ValueError(f"prediction row {row_index} bbox contains NaN or Inf")
    x, y, width, height = bbox
    if x < 0.0 or y < 0.0 or width < 0.0 or height < 0.0:
        raise ValueError(
            f"prediction row {row_index} bbox must have non-negative XYWH values, got {bbox}"
        )
    return bbox


def _canonical_rle(
    value: Any,
    *,
    row_index: int,
    expected_height: int,
    expected_width: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"prediction row {row_index} segmentation must be a COCO RLE mapping")
    if set(("size", "counts")) - set(value):
        raise ValueError(
            f"prediction row {row_index} segmentation must contain 'size' and 'counts'"
        )
    size = value["size"]
    counts = value["counts"]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"prediction row {row_index} RLE size must be [height, width]")
    if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in size):
        raise TypeError(f"prediction row {row_index} RLE size must contain integers")
    normalized_size = [int(size[0]), int(size[1])]
    if normalized_size != [expected_height, expected_width]:
        raise ValueError(
            f"prediction row {row_index} RLE size {normalized_size} does not match "
            f"GT image size {[expected_height, expected_width]}"
        )
    if not isinstance(counts, str):
        raise TypeError(
            f"prediction row {row_index} RLE counts must be a JSON string, "
            f"got {type(counts).__name__}"
        )
    rle = {"size": normalized_size, "counts": counts}
    try:
        area = float(mask_utils.area(rle))
        mask_bbox = np.asarray(mask_utils.toBbox(rle), dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"prediction row {row_index} contains invalid COCO RLE") from exc
    if not math.isfinite(area) or area < 0.0 or mask_bbox.shape != (4,):
        raise ValueError(f"prediction row {row_index} contains invalid COCO RLE geometry")
    return rle


def validate_standard_coco_rows(
    coco_gt: COCO,
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: COCOEvalContract,
    iou_types: Sequence[str],
) -> list[dict[str, Any]]:
    """Validate standard COCO rows without changing predictions."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("COCO results must be a sequence of prediction mappings")
    normalized_iou_types = tuple(iou_types)
    if not normalized_iou_types:
        raise ValueError("iou_types must not be empty")
    if len(set(normalized_iou_types)) != len(normalized_iou_types):
        raise ValueError(f"iou_types contains duplicates: {normalized_iou_types}")
    unsupported = sorted(set(normalized_iou_types) - set(SUPPORTED_IOU_TYPES))
    if unsupported:
        raise ValueError(f"unsupported COCO iou_types: {unsupported}")

    selected_image_ids = set(contract.image_ids)
    category_ids = set(contract.category_ids)
    max_predictions = contract.max_dets[-1]
    image_prediction_counts: Counter[int] = Counter()
    canonical_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"prediction row {row_index} must be a mapping")
        image_id = _require_row_int(row, "image_id", row_index=row_index)
        category_id = _require_row_int(row, "category_id", row_index=row_index)
        if image_id not in selected_image_ids:
            raise ValueError(
                f"prediction row {row_index} image_id={image_id} is not in evaluated GT IDs"
            )
        if category_id not in category_ids:
            raise ValueError(
                f"prediction row {row_index} category_id={category_id} is not in COCO GT"
            )
        if "score" not in row:
            raise ValueError(f"prediction row {row_index} is missing 'score'")
        score = float(row["score"])
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(
                f"prediction row {row_index} score must be finite and non-negative, got {score}"
            )

        bbox = None
        if "bbox" in row:
            bbox = _canonical_bbox(row["bbox"], row_index=row_index)
        elif "bbox" in normalized_iou_types:
            raise ValueError(f"prediction row {row_index} is missing standard COCO bbox")

        segmentation = None
        if "segmentation" in row:
            image_info = coco_gt.imgs.get(image_id)
            if not isinstance(image_info, Mapping):
                raise ValueError(f"COCO GT has no image metadata for image_id={image_id}")
            height = int(image_info.get("height", 0))
            width = int(image_info.get("width", 0))
            if height <= 0 or width <= 0:
                raise ValueError(f"COCO GT image_id={image_id} has invalid dimensions")
            segmentation = _canonical_rle(
                row["segmentation"],
                row_index=row_index,
                expected_height=height,
                expected_width=width,
            )
        elif "segm" in normalized_iou_types:
            raise ValueError(f"prediction row {row_index} is missing compressed COCO RLE")

        if bbox is not None and segmentation is not None:
            mask_bbox = np.asarray(mask_utils.toBbox(segmentation), dtype=np.float64)
            if not np.allclose(
                np.asarray(bbox, dtype=np.float64), mask_bbox, rtol=0.0, atol=1.0e-6
            ):
                raise ValueError(
                    f"prediction row {row_index} bbox {bbox} does not match "
                    f"RLE toBbox {mask_bbox.tolist()}"
                )

        canonical_row: dict[str, Any] = {
            "image_id": image_id,
            "category_id": category_id,
            "score": score,
        }
        if bbox is not None:
            canonical_row["bbox"] = bbox
        if segmentation is not None:
            canonical_row["segmentation"] = segmentation
        canonical_rows.append(canonical_row)
        image_prediction_counts[image_id] += 1

    excessive = {
        image_id: count
        for image_id, count in sorted(image_prediction_counts.items())
        if count > max_predictions
    }
    if excessive:
        raise ValueError(
            f"COCO results exceed maxDets={max_predictions} before evaluation: {excessive}"
        )
    return canonical_rows


def _empty_detection_coco(coco_gt: COCO) -> COCO:
    coco_dt = COCO()
    coco_dt.dataset = {
        "images": copy.deepcopy(list(coco_gt.dataset.get("images", []))),
        "categories": copy.deepcopy(list(coco_gt.dataset.get("categories", []))),
        "annotations": [],
    }
    coco_dt.createIndex()
    return coco_dt


def _load_detection_results(coco_gt: COCO, rows: list[dict[str, Any]]) -> COCO:
    if rows:
        return coco_gt.loadRes(rows)
    return _empty_detection_coco(coco_gt)


def _select_iou_indices(coco_eval: COCOeval, thresholds: Iterable[float]) -> list[int]:
    configured = np.asarray(coco_eval.params.iouThrs, dtype=np.float64)
    selected: list[int] = []
    for threshold in thresholds:
        matches = np.flatnonzero(np.isclose(configured, float(threshold), rtol=0.0, atol=1.0e-9))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one COCO IoU slice for {threshold:.2f}, found {len(matches)}"
            )
        selected.append(int(matches[0]))
    return selected


def _area_index(coco_eval: COCOeval, area_label: str) -> int:
    matches = [
        index for index, label in enumerate(coco_eval.params.areaRngLbl) if label == area_label
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one COCO area slice for {area_label!r}, found {len(matches)}"
        )
    return matches[0]


def _max_det_index(coco_eval: COCOeval, max_dets: int) -> int:
    matches = [
        index for index, value in enumerate(coco_eval.params.maxDets) if int(value) == int(max_dets)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one COCO maxDets={max_dets} slice, found {len(matches)}"
        )
    return matches[0]


def _mean_valid(values: np.ndarray) -> float:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[valid > -1.0]
    if valid.size == 0:
        return -1.0
    if not np.isfinite(valid).all():
        raise RuntimeError("COCOeval metric tensor contains NaN or Inf")
    return float(valid.mean())


def _mean_precision(
    coco_eval: COCOeval,
    *,
    max_dets: int,
    area_label: str = "all",
    iou_thresholds: Iterable[float] | None = None,
) -> float:
    precision = np.asarray(coco_eval.eval.get("precision"))
    expected_shape = (
        len(coco_eval.params.iouThrs),
        len(coco_eval.params.recThrs),
        len(coco_eval.params.catIds) if coco_eval.params.useCats else 1,
        len(coco_eval.params.areaRngLbl),
        len(coco_eval.params.maxDets),
    )
    if precision.shape != expected_shape:
        raise RuntimeError(
            f"COCOeval precision shape {precision.shape} does not match params {expected_shape}"
        )
    indices = (
        list(range(len(coco_eval.params.iouThrs)))
        if iou_thresholds is None
        else _select_iou_indices(coco_eval, iou_thresholds)
    )
    values = precision[
        indices,
        :,
        :,
        _area_index(coco_eval, area_label),
        _max_det_index(coco_eval, max_dets),
    ]
    return _mean_valid(values)


def _mean_recall(
    coco_eval: COCOeval,
    *,
    max_dets: int,
    area_label: str = "all",
) -> float:
    recall = np.asarray(coco_eval.eval.get("recall"))
    expected_shape = (
        len(coco_eval.params.iouThrs),
        len(coco_eval.params.catIds) if coco_eval.params.useCats else 1,
        len(coco_eval.params.areaRngLbl),
        len(coco_eval.params.maxDets),
    )
    if recall.shape != expected_shape:
        raise RuntimeError(
            f"COCOeval recall shape {recall.shape} does not match params {expected_shape}"
        )
    values = recall[
        :,
        :,
        _area_index(coco_eval, area_label),
        _max_det_index(coco_eval, max_dets),
    ]
    return _mean_valid(values)


def extract_coco_metrics(
    coco_eval: COCOeval,
    iou_type: str,
    *,
    max_dets: int,
) -> dict[str, float]:
    """Extract all publication metrics directly from COCOeval tensors."""

    if iou_type not in SUPPORTED_IOU_TYPES:
        raise ValueError(f"unsupported COCO iou_type: {iou_type!r}")
    prefix = iou_type
    metrics = {
        f"{prefix}_AP": _mean_precision(coco_eval, max_dets=max_dets),
        f"{prefix}_AP50": _mean_precision(coco_eval, max_dets=max_dets, iou_thresholds=(0.50,)),
        f"{prefix}_AP75": _mean_precision(coco_eval, max_dets=max_dets, iou_thresholds=(0.75,)),
    }
    high_iou_values: list[float] = []
    for threshold in (0.80, 0.85, 0.90, 0.95):
        key = int(round(threshold * 100.0))
        value = _mean_precision(coco_eval, max_dets=max_dets, iou_thresholds=(threshold,))
        metrics[f"{prefix}_AP{key}"] = value
        high_iou_values.append(value)
    metrics[f"{prefix}_AP_H"] = (
        float(np.mean(high_iou_values)) if all(value >= 0.0 for value in high_iou_values) else -1.0
    )

    size_values = {
        "s": _mean_precision(coco_eval, max_dets=max_dets, area_label="small"),
        "m": _mean_precision(coco_eval, max_dets=max_dets, area_label="medium"),
        "l": _mean_precision(coco_eval, max_dets=max_dets, area_label="large"),
    }
    for suffix, value in size_values.items():
        metrics[f"{prefix}_AP{suffix}"] = value
    metrics[f"{prefix}_AP_small"] = size_values["s"]
    metrics[f"{prefix}_AP_medium"] = size_values["m"]
    metrics[f"{prefix}_AP_large"] = size_values["l"]

    metrics[f"{prefix}_AR1"] = _mean_recall(coco_eval, max_dets=1)
    metrics[f"{prefix}_AR10"] = _mean_recall(coco_eval, max_dets=10)
    metrics[f"{prefix}_AR{max_dets}"] = _mean_recall(coco_eval, max_dets=max_dets)
    metrics[f"{prefix}_ARs"] = _mean_recall(coco_eval, max_dets=max_dets, area_label="small")
    metrics[f"{prefix}_ARm"] = _mean_recall(coco_eval, max_dets=max_dets, area_label="medium")
    metrics[f"{prefix}_ARl"] = _mean_recall(coco_eval, max_dets=max_dets, area_label="large")
    return metrics


def evaluate_standard_coco_rows(
    coco_gt: COCO,
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: COCOEvalContract,
    iou_types: Sequence[str] = SUPPORTED_IOU_TYPES,
    cocoeval_cls: type[COCOeval] = COCOeval,
) -> dict[str, float]:
    """Evaluate canonical COCO JSON rows with the official tensor contract."""

    canonical_rows = validate_standard_coco_rows(
        coco_gt,
        rows,
        contract=contract,
        iou_types=iou_types,
    )
    metrics: dict[str, float] = {}
    for iou_type in iou_types:
        if iou_type == "segm":
            rows_for_type = []
            for row in canonical_rows:
                segmentation_row = dict(row)
                segmentation_row.pop("bbox", None)
                rows_for_type.append(segmentation_row)
        else:
            rows_for_type = canonical_rows

        coco_dt = _load_detection_results(coco_gt, rows_for_type)
        coco_eval = cocoeval_cls(coco_gt, coco_dt, iouType=iou_type)
        contract.apply(coco_eval)
        coco_eval.evaluate()
        coco_eval.accumulate()
        metrics.update(
            extract_coco_metrics(
                coco_eval,
                iou_type,
                max_dets=contract.max_dets[-1],
            )
        )
    return metrics
