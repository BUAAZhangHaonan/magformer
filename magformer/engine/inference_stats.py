# -*- coding: utf-8 -*-
"""Opt-in inference instrumentation for postprocess truncation diagnostics."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from magformer.engine.coco_export import _bbox_xyxy_from_binary_mask, _mask_to_binary, _to_numpy


BUCKETS = (
    ("0-30", 0, 30),
    ("31-60", 31, 60),
    ("61-89", 61, 89),
    ("90-100", 90, 100),
    ("101+", 101, math.inf),
)


def bucket_name(count: int) -> str:
    for name, low, high in BUCKETS:
        if low <= count <= high:
            return name
    raise AssertionError(f"unreachable bucket for count={count}")


def gt_counts_by_image(coco_gt: Optional[Any]) -> Dict[int, int]:
    if coco_gt is None:
        return {}
    dataset = getattr(coco_gt, "dataset", None)
    if not isinstance(dataset, dict):
        return {}
    image_ids = [int(image["id"]) for image in dataset.get("images", []) if "id" in image]
    counts = {image_id: 0 for image_id in image_ids}
    for ann in dataset.get("annotations", []):
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        image_id = int(ann.get("image_id", -1))
        if image_id in counts:
            counts[image_id] += 1
    return counts


def count_prediction_filters(
    pred: Dict[str, Any],
    *,
    score_threshold: float,
    mask_threshold: float,
) -> Dict[str, int]:
    scores = pred.get("scores")
    masks = pred.get("masks")
    scores_arr = _to_numpy(scores) if scores is not None and len(scores) > 0 else np.zeros((0,), dtype=np.float32)
    masks_arr = _to_numpy(masks) if masks is not None and len(masks) > 0 else np.zeros((0,), dtype=np.float32)
    if masks_arr.ndim == 2:
        masks_arr = masks_arr[None, ...]

    post_score_count = 0
    post_mask_nonempty_count = 0
    for index, value in enumerate(scores_arr):
        if float(value) < float(score_threshold):
            continue
        post_score_count += 1
        if index >= len(masks_arr):
            continue
        binary = _mask_to_binary(masks_arr[index], mask_threshold=mask_threshold)
        if _bbox_xyxy_from_binary_mask(binary) is not None:
            post_mask_nonempty_count += 1

    return {
        "post_score_count": int(post_score_count),
        "post_mask_nonempty_count": int(post_mask_nonempty_count),
    }


class InferenceStatsAccumulator:
    def __init__(self, coco_gt: Optional[Any] = None) -> None:
        self._gt_counts = gt_counts_by_image(coco_gt)
        self._records: List[Dict[str, Any]] = []

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def extend_records(self, records: Iterable[Dict[str, Any]]) -> None:
        self._records.extend(dict(record) for record in records)

    def add_records(
        self,
        *,
        image_ids: Iterable[int],
        raw_stats: Iterable[Dict[str, Any]],
        predictions: Iterable[Dict[str, Any]],
        exported_rows: Iterable[Dict[str, Any]],
        score_threshold: float,
        mask_threshold: float,
    ) -> None:
        image_id_list = [int(image_id) for image_id in image_ids]
        raw_stats_list = list(raw_stats)
        pred_list = list(predictions)
        rows_by_image: Dict[int, int] = defaultdict(int)
        for row in exported_rows:
            rows_by_image[int(row["image_id"])] += 1

        if len(raw_stats_list) != len(image_id_list):
            raise ValueError(
                "inference_stats length must match evaluated image_ids: "
                f"stats={len(raw_stats_list)} image_ids={len(image_id_list)}"
            )
        if len(pred_list) != len(image_id_list):
            raise ValueError(
                "prediction length must match evaluated image_ids: "
                f"predictions={len(pred_list)} image_ids={len(image_id_list)}"
            )

        for batch_index, image_id in enumerate(image_id_list):
            raw = dict(raw_stats_list[batch_index])
            filter_counts = count_prediction_filters(
                pred_list[batch_index],
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
            )
            gt_count = int(self._gt_counts.get(image_id, 0))
            pre_topk = int(raw["pre_topk_candidate_count"])
            topk_limit = int(raw["topk_limit"])
            post_topk = int(raw["post_topk_count"])
            self._records.append(
                {
                    "image_id": image_id,
                    "gt_count": gt_count,
                    "gt_density_bucket": bucket_name(gt_count),
                    "pre_topk_candidate_count": pre_topk,
                    "topk_limit": topk_limit,
                    "post_topk_count": post_topk,
                    "topk_truncated": bool(raw.get("topk_truncated", pre_topk > post_topk)),
                    "post_score_count": filter_counts["post_score_count"],
                    "post_mask_nonempty_count": filter_counts["post_mask_nonempty_count"],
                    "exported_count": int(rows_by_image.get(image_id, 0)),
                }
            )

    def add_explicit_records(
        self,
        *,
        image_ids: Iterable[int],
        raw_stats: Iterable[Dict[str, Any]],
        filter_counts: Iterable[Dict[str, int]],
    ) -> None:
        image_id_list = [int(image_id) for image_id in image_ids]
        raw_stats_list = list(raw_stats)
        filter_count_list = list(filter_counts)
        if len(raw_stats_list) != len(image_id_list):
            raise ValueError(
                "inference_stats length must match evaluated image_ids: "
                f"stats={len(raw_stats_list)} image_ids={len(image_id_list)}"
            )
        if len(filter_count_list) != len(image_id_list):
            raise ValueError(
                "filter_counts length must match evaluated image_ids: "
                f"filter_counts={len(filter_count_list)} image_ids={len(image_id_list)}"
            )

        for batch_index, image_id in enumerate(image_id_list):
            raw = dict(raw_stats_list[batch_index])
            counts = dict(filter_count_list[batch_index])
            gt_count = int(self._gt_counts.get(image_id, 0))
            pre_topk = int(raw["pre_topk_candidate_count"])
            topk_limit = int(raw["topk_limit"])
            post_topk = int(raw["post_topk_count"])
            self._records.append(
                {
                    "image_id": image_id,
                    "gt_count": gt_count,
                    "gt_density_bucket": bucket_name(gt_count),
                    "pre_topk_candidate_count": pre_topk,
                    "topk_limit": topk_limit,
                    "post_topk_count": post_topk,
                    "topk_truncated": bool(raw.get("topk_truncated", pre_topk > post_topk)),
                    "post_score_count": int(counts["post_score_count"]),
                    "post_mask_nonempty_count": int(counts["post_mask_nonempty_count"]),
                    "exported_count": int(counts["exported_count"]),
                }
            )

    def payload(self) -> Dict[str, Any]:
        buckets: Dict[str, Dict[str, int]] = {
            name: {
                "images": 0,
                "gt": 0,
                "pre_topk_candidates": 0,
                "post_topk": 0,
                "post_score": 0,
                "post_mask_nonempty": 0,
                "exported": 0,
                "topk_truncated_images": 0,
            }
            for name, _, _ in BUCKETS
        }
        for record in self._records:
            bucket = buckets[record["gt_density_bucket"]]
            bucket["images"] += 1
            bucket["gt"] += int(record["gt_count"])
            bucket["pre_topk_candidates"] += int(record["pre_topk_candidate_count"])
            bucket["post_topk"] += int(record["post_topk_count"])
            bucket["post_score"] += int(record["post_score_count"])
            bucket["post_mask_nonempty"] += int(record["post_mask_nonempty_count"])
            bucket["exported"] += int(record["exported_count"])
            if bool(record["topk_truncated"]):
                bucket["topk_truncated_images"] += 1

        total_images = len(self._records)
        return {
            "summary": {
                "total_images": total_images,
                "topk_truncated_images": sum(1 for row in self._records if row["topk_truncated"]),
                "score_filtered_images": sum(
                    1 for row in self._records if int(row["post_score_count"]) < int(row["post_topk_count"])
                ),
                "mask_empty_filtered_images": sum(
                    1 for row in self._records if int(row["post_mask_nonempty_count"]) < int(row["post_score_count"])
                ),
                "buckets": buckets,
            },
            "images": self._records,
        }

    def dump(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out
