from __future__ import annotations

import numpy as np
import pytest
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from magformer.engine.coco_json_eval import (
    build_coco_eval_contract,
    validate_standard_coco_rows,
)
from magformer.engine.evaluator import COCOEvaluator
from tools.evaluate import rescale_predictions_to_original


def _rle(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def _geometry_coco(height: int, width: int) -> COCO:
    coco = COCO()
    coco.dataset = {
        "images": [{"id": 7, "file_name": "seven.png", "height": height, "width": width}],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }
    coco.createIndex()
    return coco


def test_noninteger_mask_rescale_derives_bbox_from_final_8_to_10_mask() -> None:
    source_mask = np.zeros((8, 8), dtype=np.uint8)
    source_mask[1:4, 2:5] = 1
    source_rle = _rle(source_mask)
    source_bbox_xywh = mask_utils.toBbox(source_rle).tolist()
    source_bbox_xyxy = [
        source_bbox_xywh[0],
        source_bbox_xywh[1],
        source_bbox_xywh[0] + source_bbox_xywh[2],
        source_bbox_xywh[1] + source_bbox_xywh[3],
    ]
    scaled_old_bbox = [value * 1.25 for value in source_bbox_xyxy]

    [rescaled] = rescale_predictions_to_original(
        [
            {
                "image_id": 7,
                "category_id": 1,
                "score": 0.9,
                "bbox": source_bbox_xyxy,
                "mask": source_rle,
            }
        ],
        original_sizes={7: (10, 10)},
        target_size=(8, 8),
    )

    final_bbox_xywh = [float(value) for value in mask_utils.toBbox(rescaled["mask"])]
    expected_xyxy = [
        final_bbox_xywh[0],
        final_bbox_xywh[1],
        final_bbox_xywh[0] + final_bbox_xywh[2],
        final_bbox_xywh[1] + final_bbox_xywh[3],
    ]
    assert rescaled["bbox"] == expected_xyxy
    assert not np.allclose(rescaled["bbox"], scaled_old_bbox, rtol=0.0, atol=1.0e-12)

    coco_gt = _geometry_coco(10, 10)
    evaluator = COCOEvaluator(coco_gt=coco_gt, iou_types=["bbox", "segm"])
    evaluator.update([rescaled], image_ids=[7])
    standard_rows = evaluator.to_coco_results()
    canonical_rows = validate_standard_coco_rows(
        coco_gt,
        standard_rows,
        contract=build_coco_eval_contract(coco_gt),
        iou_types=("bbox", "segm"),
    )
    assert canonical_rows[0]["bbox"] == final_bbox_xywh


def test_standard_segmentation_rescale_keeps_standard_xywh_geometry() -> None:
    source_mask = np.zeros((8, 8), dtype=np.uint8)
    source_mask[1:4, 2:5] = 1
    [rescaled] = rescale_predictions_to_original(
        [
            {
                "image_id": 7,
                "category_id": 1,
                "score": 0.9,
                "bbox": [2.0, 1.0, 3.0, 3.0],
                "segmentation": _rle(source_mask),
            }
        ],
        original_sizes={7: (10, 10)},
        target_size=(8, 8),
    )

    assert "mask" not in rescaled
    assert rescaled["bbox"] == pytest.approx(mask_utils.toBbox(rescaled["segmentation"]))


@pytest.mark.parametrize("geometry_key", ["mask", "segmentation"])
def test_empty_rescaled_mask_has_zero_box(geometry_key: str) -> None:
    empty_mask = np.zeros((8, 8), dtype=np.uint8)
    [rescaled] = rescale_predictions_to_original(
        [
            {
                "image_id": 7,
                "category_id": 1,
                "score": 0.1,
                "bbox": [0.0, 0.0, 0.0, 0.0],
                geometry_key: _rle(empty_mask),
            }
        ],
        original_sizes={7: (10, 10)},
        target_size=(8, 8),
    )

    assert rescaled["bbox"] == [0.0, 0.0, 0.0, 0.0]
