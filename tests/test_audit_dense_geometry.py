from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from tools.audit_dense_geometry import run_audit


def _rle(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def _bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    return [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


def _square(x0: int, y0: int, size: int = 4) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[y0: y0 + size, x0: x0 + size] = 1
    return mask


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_audit_classifies_fp_types_and_dense_buckets(tmp_path: Path) -> None:
    image1_gt = [_square(2, 2), _square(20, 2)]
    dense_gt = [_square((idx % 8) * 4, (idx // 8) * 4, size=2) for idx in range(31)]
    annotations = []
    ann_id = 1
    for image_id, masks in [(1, image1_gt), (2, dense_gt)]:
        for mask in masks:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": _bbox(mask),
                    "area": float(mask.sum()),
                    "segmentation": _rle(mask),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    gt_path = _write_json(
        tmp_path / "gt.json",
        {
            "images": [
                {"id": 1, "file_name": "one.png", "height": 32, "width": 32},
                {"id": 2, "file_name": "dense.png", "height": 32, "width": 32},
            ],
            "annotations": annotations,
            "categories": [{"id": 1, "name": "component"}],
        },
    )

    tp = image1_gt[0]
    duplicate = image1_gt[0]
    low_iou = _square(4, 2)
    background = _square(12, 20)
    pred_path = _write_json(
        tmp_path / "pred.json",
        [
            {"image_id": 1, "category_id": 1, "score": 0.99, "bbox": _bbox(tp), "segmentation": _rle(tp)},
            {"image_id": 1, "category_id": 1, "score": 0.80, "bbox": _bbox(duplicate), "segmentation": _rle(duplicate)},
            {"image_id": 1, "category_id": 1, "score": 0.70, "bbox": _bbox(low_iou), "segmentation": _rle(low_iou)},
            {"image_id": 1, "category_id": 1, "score": 0.60, "bbox": _bbox(background), "segmentation": _rle(background)},
        ],
    )

    summary = run_audit(gt_path, pred_path, name="unit", output_dir=tmp_path / "out", top_k=2)

    image_one = summary["images"]["1"]
    assert image_one["gt_count"] == 2
    assert image_one["pred_count"] == 4
    assert image_one["mask"]["tp"] == 1
    assert image_one["mask"]["fn"] == 1
    assert image_one["mask"]["duplicate_fp"] == 1
    assert image_one["mask"]["low_iou_fp"] == 1
    assert image_one["mask"]["background_fp"] == 1
    assert image_one["mask"]["tp_mean_iou"] == pytest.approx(1.0)
    assert summary["buckets"]["0-30"]["images"] == 1
    assert summary["buckets"]["31-60"]["images"] == 1
    assert summary["max_dets"]["postprocess_pretopk"] == "postprocess-pretopk unavailable"
    assert summary["max_dets"]["predictions_per_image"]["max"] == 4
    assert summary["geometry"]["matched_tp_mask_iou"]["mean"] == pytest.approx(1.0)
    assert summary["geometry"]["low_iou_fp_max_iou"]["count"] == 1
    assert (tmp_path / "out" / "unit_audit.json").exists()
    assert (tmp_path / "out" / "unit_audit.md").exists()
