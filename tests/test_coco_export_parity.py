import numpy as np
import pytest

from magformer.engine.coco_export import (
    internal_instances_to_coco_results,
    outputs_to_coco_instances,
    predictions_to_coco_instances,
)
from magformer.engine.trainer import Trainer


def test_predictions_to_coco_instances_filters_and_maps_consistently():
    masks = np.zeros((2, 4, 4), dtype=np.float32)
    masks[0, 1:3, 1:4] = 0.9
    masks[1, 0:2, 0:2] = 0.8

    predictions = [
        {
            "scores": np.array([0.7, 0.01], dtype=np.float32),
            "category_ids": np.array([0, 0], dtype=np.int64),
            "masks": masks,
        }
    ]

    rows = predictions_to_coco_instances(
        predictions=predictions,
        image_ids=[123],
        score_threshold=0.05,
        mask_threshold=0.5,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["image_id"] == 123
    assert row["category_id"] == 1
    assert abs(row["score"] - 0.7) < 1e-6
    assert row["bbox"] == [1.0, 1.0, 4.0, 3.0]


def test_outputs_wrapper_matches_direct_predictions_path():
    predictions = [
        {
            "scores": [0.8],
            "category_ids": [0],
            "masks": [np.array([[0, 1], [1, 1]], dtype=np.float32)],
        }
    ]
    outputs = {"predictions": predictions}

    a = predictions_to_coco_instances(predictions, image_ids=[7], score_threshold=0.05)
    b = outputs_to_coco_instances(outputs, image_ids=[7], score_threshold=0.05)

    assert len(a) == len(b) == 1
    assert a[0]["image_id"] == b[0]["image_id"] == 7
    assert a[0]["category_id"] == b[0]["category_id"] == 1
    assert a[0]["bbox"] == b[0]["bbox"]


def test_bbox_only_export_skips_segmentation_rle(monkeypatch):
    predictions = [
        {
            "scores": [0.8],
            "category_ids": [0],
            "masks": [np.array([[0, 1], [1, 1]], dtype=np.float32)],
        }
    ]

    def _fail_encode(_mask):
        raise AssertionError("RLE encoding should not run for bbox-only export")

    monkeypatch.setattr("magformer.engine.coco_export._encode_mask_rle", _fail_encode)

    rows = predictions_to_coco_instances(
        predictions,
        image_ids=[7],
        score_threshold=0.05,
        include_segmentation=False,
    )
    wrapped_rows = outputs_to_coco_instances(
        {"predictions": predictions},
        image_ids=[7],
        score_threshold=0.05,
        include_segmentation=False,
    )

    assert rows[0]["bbox"] == [0.0, 0.0, 2.0, 2.0]
    assert "mask" not in rows[0]
    assert "mask" not in wrapped_rows[0]


def test_backmap_internal_rows_convert_to_standard_coco_results():
    internal_rows = [
        {
            "image_id": 123,
            "category_id": 1,
            "score": 0.9,
            "bbox": [2.0, 3.0, 8.0, 13.0],
            "mask": {"size": [16, 16], "counts": "P1370000l0"},
        }
    ]

    coco_rows = internal_instances_to_coco_results(internal_rows, iou_types=["bbox", "segm"])

    assert coco_rows == [
        {
            "image_id": 123,
            "category_id": 1,
            "score": 0.9,
            "bbox": [2.0, 3.0, 6.0, 10.0],
            "segmentation": {"size": [16, 16], "counts": "P1370000l0"},
        }
    ]
    assert "mask" not in coco_rows[0]


def test_backmap_bbox_only_coco_results_omit_segmentation():
    internal_rows = [
        {
            "image_id": 123,
            "category_id": 1,
            "score": 0.9,
            "bbox": [2.0, 3.0, 8.0, 13.0],
            "mask": {"size": [16, 16], "counts": "P1370000l0"},
        }
    ]

    coco_rows = internal_instances_to_coco_results(internal_rows, iou_types=["bbox"])

    assert coco_rows == [
        {
            "image_id": 123,
            "category_id": 1,
            "score": 0.9,
            "bbox": [2.0, 3.0, 6.0, 10.0],
        }
    ]
    assert "mask" not in coco_rows[0]
    assert "segmentation" not in coco_rows[0]


def test_outputs_wrapper_requires_predictions_key():
    with pytest.raises(KeyError, match="predictions"):
        outputs_to_coco_instances({}, image_ids=[7], score_threshold=0.05)


def test_predictions_to_coco_instances_rejects_length_mismatch():
    predictions = [
        {
            "scores": np.array([0.8, 0.7], dtype=np.float32),
            "category_ids": np.array([0], dtype=np.int64),
            "masks": np.ones((2, 4, 4), dtype=np.float32),
        }
    ]

    with pytest.raises(ValueError, match="length mismatch"):
        predictions_to_coco_instances(predictions, image_ids=[7], score_threshold=0.05)


def test_predictions_to_coco_instances_rejects_nonfinite_masks():
    masks = np.ones((1, 4, 4), dtype=np.float32)
    masks[0, 0, 0] = np.nan
    predictions = [
        {
            "scores": np.array([0.8], dtype=np.float32),
            "category_ids": np.array([0], dtype=np.int64),
            "masks": masks,
        }
    ]

    with pytest.raises(ValueError, match="NaN or Inf"):
        predictions_to_coco_instances(predictions, image_ids=[7], score_threshold=0.05)


def test_trainer_convert_to_coco_format_delegates_to_shared_export(monkeypatch):
    called = {}

    def _fake_shared_export(outputs, image_ids=None, score_threshold=0.05, mask_threshold=0.5, category_offset=1):
        called["outputs"] = outputs
        called["image_ids"] = image_ids
        called["score_threshold"] = score_threshold
        called["mask_threshold"] = mask_threshold
        called["category_offset"] = category_offset
        return [{"image_id": 9, "category_id": 1, "score": 0.9, "bbox": [0.0, 0.0, 1.0, 1.0], "mask": np.ones((2, 2), dtype=np.uint8)}]

    monkeypatch.setattr("magformer.engine.trainer.outputs_to_coco_instances", _fake_shared_export)

    trainer = Trainer.__new__(Trainer)
    outputs = {"predictions": [{"scores": [0.9], "category_ids": [0], "masks": [np.ones((2, 2), dtype=np.float32)]}]}
    image_ids = [42]

    rows = trainer._convert_to_coco_format(outputs, image_ids)

    assert len(rows) == 1
    assert rows[0]["image_id"] == 9
    assert called["outputs"] is outputs
    assert called["image_ids"] == image_ids
    assert called["score_threshold"] == 0.05
    assert called["mask_threshold"] == 0.5
    assert called["category_offset"] == 1
