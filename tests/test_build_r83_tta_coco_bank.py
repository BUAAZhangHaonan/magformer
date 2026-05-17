from __future__ import annotations

import importlib

import numpy as np
from pycocotools import mask as coco_mask


def _r83():
    return importlib.import_module("tools.build_r83_tta_coco_bank")


def test_candidate_to_coco_annotation_writes_valid_rle_and_xywh_bbox() -> None:
    r82 = importlib.import_module("tools.diagnose_r82_tta_candidate_bank")
    r83 = _r83()
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3:7] = 1
    candidate = r82.Candidate(
        quality=0.95,
        category_id=1,
        bbox=[3.0, 2.0, 7.0, 6.0],
        mask=mask,
        source_view="scale1_noflip",
    )

    annotation = r83.candidate_to_coco_annotation(
        candidate,
        annotation_id=11,
        image_id=123,
        image_height=8,
        image_width=8,
    )

    assert annotation["id"] == 11
    assert annotation["image_id"] == 123
    assert annotation["category_id"] == 1
    assert annotation["bbox"] == [3.0, 2.0, 4.0, 4.0]
    assert annotation["area"] == 16
    assert annotation["score"] == 0.95
    assert annotation["iscrowd"] == 0
    decoded = coco_mask.decode(annotation["segmentation"])
    np.testing.assert_array_equal(decoded, mask)


def test_select_training_candidates_applies_score_area_and_fill_filters() -> None:
    r82 = importlib.import_module("tools.diagnose_r82_tta_candidate_bank")
    r83 = _r83()
    good = np.zeros((12, 12), dtype=np.uint8)
    good[1:7, 1:7] = 1
    tiny = np.zeros((12, 12), dtype=np.uint8)
    tiny[1:4, 1:4] = 1
    sparse = np.zeros((20, 20), dtype=np.uint8)
    sparse[1:6, 1:6] = 1
    low_score = good.copy()

    candidates = [
        r82.Candidate(0.95, 1, [1.0, 1.0, 7.0, 7.0], good, "good"),
        r82.Candidate(0.99, 1, [1.0, 1.0, 4.0, 4.0], tiny, "tiny"),
        r82.Candidate(0.99, 1, [0.0, 0.0, 20.0, 20.0], sparse, "sparse"),
        r82.Candidate(0.89, 1, [1.0, 1.0, 7.0, 7.0], low_score, "low"),
    ]

    kept, stats = r83.select_training_candidates(
        candidates,
        min_score=0.90,
        min_mask_area=20,
        min_fill_ratio=0.1,
    )

    assert [item.source_view for item in kept] == ["good"]
    assert stats == {
        "input": 4,
        "kept": 1,
        "dropped_score": 1,
        "dropped_mask_area": 1,
        "dropped_fill_ratio": 1,
    }
