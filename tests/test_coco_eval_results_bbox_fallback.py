from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from baselines.coco_eval_results import evaluate_coco_results


def _write_simple_coco(path: Path) -> None:
    payload = {
        "images": [{"id": 1, "file_name": "000001.png", "width": 16, "height": 16}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [4, 5, 6, 7],
                "area": 42,
                "iscrowd": 0,
                "segmentation": [[4, 5, 10, 5, 10, 12, 4, 12]],
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_coco_results_recovers_bbox_from_segmentation_when_bbox_is_zero(tmp_path: Path) -> None:
    ann_path = tmp_path / "instances_val.json"
    _write_simple_coco(ann_path)

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5:12, 4:10] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")

    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "score": 0.9,
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "segmentation": rle,
                }
            ]
        ),
        encoding="utf-8",
    )

    metrics = evaluate_coco_results(ann_file=ann_path, results_json=results_path, iteration=7)

    assert metrics["iteration"] == 7
    assert metrics["bbox/AP50"] > 99.0
    assert metrics["segm/AP50"] > 99.0
