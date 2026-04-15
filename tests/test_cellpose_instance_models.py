from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "baselines" / "cellpose_instance_models.py"
    spec = importlib.util.spec_from_file_location("cellpose_instance_models", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_instance_map_to_cellpose_targets_produces_flow_and_cellprob() -> None:
    mod = _load_module()
    instance_map = np.zeros((16, 16), dtype=np.int32)
    instance_map[2:6, 2:6] = 1
    instance_map[9:13, 9:14] = 2

    targets = mod.instance_map_to_cellpose_targets(instance_map)

    assert targets["cellprob"].shape == (16, 16)
    assert targets["flow"].shape == (2, 16, 16)
    assert targets["cellprob"].dtype == np.float32
    assert targets["flow"].dtype == np.float32
    assert targets["cellprob"][3, 3] == 1.0
    assert targets["cellprob"][0, 0] == 0.0
    assert np.any(np.abs(targets["flow"]) > 0.0)


def test_follow_flows_and_scores_round_trip_separates_instances() -> None:
    mod = _load_module()
    instance_map = np.zeros((24, 24), dtype=np.int32)
    instance_map[3:9, 3:9] = 1
    instance_map[14:20, 14:21] = 2
    targets = mod.instance_map_to_cellpose_targets(instance_map)

    masks, scores, category_ids = mod.predictions_from_logits(
        flow_logits=targets["flow"] * 5.0,
        cellprob_logits=np.where(targets["cellprob"] > 0, 8.0, -8.0).astype(np.float32),
        min_area=5,
        score_threshold=0.05,
        mask_threshold=0.5,
    )

    assert len(masks) == 2
    assert scores.shape == (2,)
    assert category_ids.tolist() == [0, 0]
    assert all(mask.dtype == np.uint8 for mask in masks)
    assert scores[0] <= 1.0 and scores[1] <= 1.0
    assert scores[0] >= 0.5 and scores[1] >= 0.5


def test_evaluate_results_serializes_mask_rows_for_coco_eval(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools")

    mod = _load_module()
    dataset_root = tmp_path / "ecc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)

    image_name = "val_000001.png"
    Image.new("RGB", (16, 16), color=(12, 34, 56)).save(dataset_root / "images" / "val" / image_name)
    payload = {
        "images": [{"id": 1, "file_name": image_name, "width": 16, "height": 16}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[2, 2, 7, 2, 7, 7, 2, 7]],
                "area": 25,
                "bbox": [2, 2, 5, 5],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (dataset_root / "annotations" / "instances_val.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = [
        {
            "image_id": 1,
            "category_id": 1,
            "score": 0.9,
            "bbox": [2.0, 2.0, 5.0, 5.0],
            "mask": np.pad(np.ones((5, 5), dtype=np.uint8), ((2, 9), (2, 9))),
        }
    ]
    metrics = mod.evaluate_results(
        dataset_root=dataset_root,
        eval_split="val",
        rows=rows,
        image_size=16,
        iteration=1,
    )

    assert "segm/AP" in metrics
