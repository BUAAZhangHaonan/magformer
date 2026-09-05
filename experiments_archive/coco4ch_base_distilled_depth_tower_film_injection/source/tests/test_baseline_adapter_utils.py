from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_annotations_to_instance_targets_preserves_order_and_decodes_masks() -> None:
    pytest.importorskip("pycocotools")

    from pycocotools import mask as mask_utils

    from baselines.baseline_adapter_utils import annotations_to_instance_targets

    mask_a = np.zeros((8, 8), dtype=np.uint8)
    mask_a[1:4, 1:4] = 1
    rle_b = mask_utils.encode(np.asfortranarray(np.pad(np.ones((2, 3), dtype=np.uint8), ((4, 2), (2, 3)))))
    rle_b["counts"] = rle_b["counts"].decode("utf-8")

    annotations = [
        {
            "id": 17,
            "category_id": 5,
            "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
        },
        {
            "id": 18,
            "category_id": 9,
            "segmentation": rle_b,
        },
    ]

    targets = annotations_to_instance_targets(annotations, height=8, width=8)

    assert targets["instance_map"].dtype == np.int32
    assert targets["instance_map"][1, 1] == 1
    assert targets["instance_map"][4, 2] == 2
    assert targets["category_ids"].tolist() == [5, 9]
    assert len(targets["masks"]) == 2
    assert targets["masks"][0].dtype == np.uint8
    assert targets["masks"][1].dtype == np.uint8


def test_resolve_instance_scores_prefers_explicit_scores_and_falls_back_to_masks() -> None:
    from baselines.baseline_adapter_utils import resolve_instance_scores

    explicit = resolve_instance_scores([0.25, 0.75], masks=[np.ones((2, 2)), np.zeros((2, 2))])
    fallback = resolve_instance_scores(None, masks=[np.ones((2, 2)), np.array([[0.0, 0.5], [1.0, 0.5]], dtype=np.float32)])

    assert explicit.dtype == np.float32
    assert explicit.tolist() == [0.25, 0.75]
    assert fallback.tolist() == [1.0, 0.5]


def test_binary_masks_to_coco_rows_thresholds_and_maps_categories() -> None:
    from baselines.baseline_adapter_utils import binary_masks_to_coco_rows

    masks = np.zeros((2, 4, 4), dtype=np.float32)
    masks[0, 1:3, 1:4] = 0.9
    masks[1, 0:2, 0:2] = 0.4

    rows = binary_masks_to_coco_rows(
        image_id=42,
        masks=masks,
        scores=[0.8, 0.01],
        category_ids=[0, 1],
        score_threshold=0.05,
        mask_threshold=0.5,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["image_id"] == 42
    assert row["category_id"] == 1
    assert abs(row["score"] - 0.8) < 1e-6
    assert row["bbox"] == [1.0, 1.0, 3.0, 2.0]
    assert row["mask"].dtype == np.uint8


def test_write_baseline_run_artifacts_creates_standard_files(tmp_path: Path) -> None:
    from baselines.baseline_adapter_utils import write_baseline_run_artifacts

    out_dir = tmp_path / "run"
    rows = [{"image_id": 7, "category_id": 1, "score": 0.9, "bbox": [1.0, 2.0, 3.0, 4.0], "mask": np.ones((2, 2), dtype=np.uint8)}]
    artifacts = write_baseline_run_artifacts(
        out_dir,
        coco_rows=rows,
        metrics={"segm_AP": 0.5},
        metadata={"model_id": "toy", "status": "done"},
        last_checkpoint="model_final.pth",
        wall_time_sec=12.5,
        trainable_params=1234,
    )

    assert artifacts["coco_instances_results"].exists()
    assert artifacts["metrics_cocoeval"].exists()
    assert artifacts["metadata"].exists()
    assert artifacts["last_checkpoint"].exists()
    assert artifacts["wall_time_sec"].exists()
    assert artifacts["params_trainable"].exists()

    coco_rows = json.loads(artifacts["coco_instances_results"].read_text(encoding="utf-8"))
    metrics = json.loads(artifacts["metrics_cocoeval"].read_text(encoding="utf-8"))
    metadata = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))

    assert coco_rows[0]["image_id"] == 7
    assert "segmentation" in coco_rows[0]
    assert "mask" not in coco_rows[0]
    assert metrics["segm_AP"] == 0.5
    assert metadata["model_id"] == "toy"
    assert metadata["status"] == "done"
    assert artifacts["last_checkpoint"].read_text(encoding="utf-8").strip() == "model_final.pth"
    assert artifacts["wall_time_sec"].read_text(encoding="utf-8").strip() == "12.5"
    assert artifacts["params_trainable"].read_text(encoding="utf-8").strip() == "1234"
