from __future__ import annotations

import importlib
import math

import numpy as np


def _r82():
    return importlib.import_module("tools.diagnose_r82_tta_candidate_bank")


def test_union_clusters_same_class_by_bbox_iou_and_keeps_best_mask() -> None:
    r82 = _r82()
    mask_a = np.zeros((8, 8), dtype=np.uint8)
    mask_a[1:7, 1:7] = 1
    mask_b = np.zeros((8, 8), dtype=np.uint8)
    mask_b[1:4, 1:4] = 1
    mask_c = np.zeros((8, 8), dtype=np.uint8)
    mask_c[1:7, 1:7] = 1

    candidates = [
        r82.Candidate(0.70, 1, [1.0, 1.0, 7.0, 7.0], mask_a, "s1"),
        r82.Candidate(0.90, 1, [1.2, 1.1, 6.8, 6.9], mask_b, "s2"),
        r82.Candidate(0.95, 2, [1.0, 1.0, 7.0, 7.0], mask_c, "other-class"),
    ]

    merged = r82.union_cluster_candidates(
        candidates,
        mask_iou_threshold=0.55,
        bbox_iou_threshold=0.75,
    )

    assert len(merged) == 2
    class_one = next(item for item in merged if item.category_id == 1)
    assert class_one.quality == 0.90
    assert class_one.source_view == "s2"
    np.testing.assert_array_equal(class_one.mask, mask_b)


def test_bucket_summary_values_are_finite_for_tiny_and_bottom_rows() -> None:
    r82 = _r82()
    gt_by_image = {
        10: {
            "image": {"id": 10, "width": 8, "height": 8},
            "all": [
                {"id": 1, "image_id": 10, "bbox": [1, 1, 2, 2], "_area": 4.0},
            ],
            "tiny_area_le_256": [
                {"id": 1, "image_id": 10, "bbox": [1, 1, 2, 2], "_area": 4.0},
            ],
            "bottom20_area": [
                {"id": 1, "image_id": 10, "bbox": [1, 1, 2, 2], "_area": 4.0},
            ],
        }
    }
    r78_by_image = {10: {"r78_bucket": "normal", "file_name": "x.png"}}
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    record = {
        "image_id": 10,
        "target_size": (8, 8),
        "candidates": [r82.Candidate(0.2, 1, [1.0, 1.0, 3.0, 3.0], mask, "v")],
        "kept": [r82.Candidate(0.2, 1, [1.0, 1.0, 3.0, 3.0], mask, "v")],
    }

    summary = r82.summarize_candidate_bank_buckets(
        [record],
        gt_by_image=gt_by_image,
        r78_by_image=r78_by_image,
        bottom20_threshold=4.0,
        target_ann_path="ann.json",
        r78_stats_path="stats.json",
    )

    for bucket in summary["buckets"].values():
        for value in bucket.values():
            if isinstance(value, float):
                assert math.isfinite(value)

    assert summary["buckets"]["tiny_area_le_256"]["candidate_gt_coverage_iou50"] == 1.0
    assert summary["buckets"]["bottom20_area"]["kept_gt_coverage_iou75"] == 1.0
