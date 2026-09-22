import numpy as np

from magformer.engine.coco_export import predictions_to_coco_instances


def test_empty_mask_is_dropped_by_default():
    predictions = [
        {
            "scores": np.array([0.9], dtype=np.float32),
            "category_ids": np.array([0], dtype=np.int64),
            # logits/probs that will become empty at threshold=0.5
            "masks": np.array([np.full((8, 8), 0.1, dtype=np.float32)]),
        }
    ]

    rows = predictions_to_coco_instances(
        predictions=predictions,
        image_ids=[1],
        score_threshold=0.05,
        mask_threshold=0.5,
    )
    assert rows == []


def test_empty_mask_fallback_requires_explicit_opt_in():
    predictions = [
        {
            "scores": np.array([0.9], dtype=np.float32),
            "category_ids": np.array([0], dtype=np.int64),
            "masks": np.array([np.full((8, 8), 0.1, dtype=np.float32)]),
        }
    ]

    rows = predictions_to_coco_instances(
        predictions=predictions,
        image_ids=[1],
        score_threshold=0.05,
        mask_threshold=0.5,
        allow_empty_fallback=True,
        empty_fallback_ratio=0.1,
    )
    assert len(rows) == 1
    assert rows[0]["image_id"] == 1
    assert rows[0]["category_id"] == 1
