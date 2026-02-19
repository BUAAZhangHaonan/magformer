from __future__ import annotations

import json
from pathlib import Path


def test_ucn_coco_results_json_is_cocoeval_readable(tmp_path: Path) -> None:
    """
    UCN baseline exports `coco_instances_results.json` for COCOeval.
    This test ensures our RLE encoding + json writing path is compatible.
    """
    from baselines.ucn_coco_utils import coco_eval_stats, encode_binary_mask_rle

    gt = {
        "images": [{"id": 1, "file_name": "dummy.png", "width": 10, "height": 10}],
        "categories": [{"id": 1, "name": "component"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 5, 5],
                "area": 25,
                "iscrowd": 0,
                "segmentation": [[1, 1, 6, 1, 6, 6, 1, 6]],
            }
        ],
    }
    gt_path = tmp_path / "instances_val.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")

    import numpy as np

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:6, 1:6] = 1
    rle = encode_binary_mask_rle(mask)

    dt = [{"image_id": 1, "category_id": 1, "segmentation": rle, "bbox": [1, 1, 5, 5], "score": 1.0}]
    dt_path = tmp_path / "coco_instances_results.json"
    dt_path.write_text(json.dumps(dt), encoding="utf-8")

    stats = coco_eval_stats(str(gt_path), str(dt_path), iou_type="segm")
    assert 0.99 <= float(stats["AP"]) <= 1.0

