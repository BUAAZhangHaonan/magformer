from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from tools.build_pseudo_bank import BankConfig, DynamicBucketRule, build_pseudo_bank


def _rle(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def _bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    return [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


def _square(x0: int, y0: int, size: int = 4) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[y0 : y0 + size, x0 : x0 + size] = 1
    return mask


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _coco(images: list[dict], annotations: list[dict] | None = None) -> dict:
    return {
        "images": images,
        "annotations": annotations or [],
        "categories": [{"id": 1, "name": "component"}],
    }


def _ann(ann_id: int, image_id: int, mask: np.ndarray) -> dict:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": _bbox(mask),
        "area": float(mask.sum()),
        "segmentation": _rle(mask),
        "iscrowd": 0,
    }


def _pred(image_id: int, score: float, mask: np.ndarray, *, category_id: int = 1) -> dict:
    return {
        "image_id": image_id,
        "category_id": category_id,
        "score": score,
        "bbox": _bbox(mask),
        "segmentation": _rle(mask),
    }


def test_build_pseudo_bank_filters_and_writes_mask_consistent_coco(tmp_path: Path) -> None:
    target_images = [{"id": 1, "file_name": "one.png", "height": 32, "width": 32}]
    source_path = _write_json(tmp_path / "source.json", _coco([{"id": 99, "file_name": "source.png", "height": 32, "width": 32}]))
    target_path = _write_json(tmp_path / "target.json", _coco(target_images))
    gt_mask = _square(2, 2, 4)
    hidden_gt_path = _write_json(tmp_path / "hidden_gt.json", _coco(target_images, [_ann(1, 1, gt_mask), _ann(2, 1, _square(20, 2, 4))]))
    predictions_path = _write_json(
        tmp_path / "pred.json",
        [
            _pred(1, 0.99, gt_mask),
            _pred(1, 0.93, _square(10, 10, 4)),
            _pred(1, 0.98, _square(16, 16, 1)),
        ],
    )

    summary = build_pseudo_bank(
        predictions_json=predictions_path,
        source_coco_json=source_path,
        target_coco_json=target_path,
        output_json=tmp_path / "bank.json",
        metrics_json=tmp_path / "metrics.json",
        metrics_md=tmp_path / "metrics.md",
        config=BankConfig(score_min=0.94, fill_min=0.55, area_min=10.0, bbox_min_side=2.0, bbox_max_side=20.0, per_image_topk=1),
        hidden_gt_json=hidden_gt_path,
        name="unit_simple",
    )

    bank = json.loads((tmp_path / "bank.json").read_text(encoding="utf-8"))
    assert bank["images"] == target_images
    assert bank["categories"] == [{"id": 1, "name": "component"}]
    assert len(bank["annotations"]) == 1
    out_ann = bank["annotations"][0]
    assert out_ann["score"] == pytest.approx(0.99)
    assert out_ann["bbox"] == [2.0, 2.0, 4.0, 4.0]
    assert out_ann["area"] == pytest.approx(16.0)
    rle = dict(out_ann["segmentation"])
    rle["counts"] = rle["counts"].encode("ascii")
    assert float(mask_utils.area(rle)) == pytest.approx(out_ann["area"])
    assert [float(v) for v in mask_utils.toBbox(rle).tolist()] == out_ann["bbox"]
    assert summary["totals"]["kept"] == 1
    assert summary["quality"]["mask"]["P50"] == pytest.approx(1.0)
    assert summary["quality"]["mask"]["R50"] == pytest.approx(0.5)
    assert summary["quality"]["mask"]["P75"] == pytest.approx(1.0)
    assert summary["quality"]["mask"]["R75"] == pytest.approx(0.5)
    assert (tmp_path / "metrics.md").read_text(encoding="utf-8").startswith("# Pseudo Bank Quality: unit_simple")


def test_dynamic_bucket_rules_override_thresholds_by_raw_prediction_count(tmp_path: Path) -> None:
    target_images = [
        {"id": 1, "file_name": "sparse.png", "height": 32, "width": 32},
        {"id": 2, "file_name": "dense.png", "height": 32, "width": 32},
    ]
    source_path = _write_json(tmp_path / "source.json", _coco(target_images))
    target_path = _write_json(tmp_path / "target.json", _coco(target_images))
    sparse = _square(2, 2, 4)
    dense_hi = _square(8, 8, 4)
    dense_low = _square(16, 16, 4)
    predictions_path = _write_json(
        tmp_path / "pred.json",
        [
            _pred(1, 0.95, sparse),
            _pred(2, 0.98, dense_hi),
            _pred(2, 0.95, dense_low),
            _pred(2, 0.94, _square(22, 22, 4)),
        ],
    )

    summary = build_pseudo_bank(
        predictions_json=predictions_path,
        source_coco_json=source_path,
        target_coco_json=target_path,
        output_json=tmp_path / "bank.json",
        metrics_json=tmp_path / "metrics.json",
        metrics_md=tmp_path / "metrics.md",
        config=BankConfig(score_min=0.94, fill_min=0.55, area_min=10.0),
        dynamic_rules=[DynamicBucketRule(low=3, high=10, score_min=0.97, fill_min=0.55, area_min=10.0)],
        name="unit_dynamic",
    )

    bank = json.loads((tmp_path / "bank.json").read_text(encoding="utf-8"))
    kept_by_image = {ann["image_id"]: ann["score"] for ann in bank["annotations"]}
    assert kept_by_image == {1: pytest.approx(0.95), 2: pytest.approx(0.98)}
    assert summary["totals"]["raw_predictions"] == 4
    assert summary["totals"]["kept"] == 2
    assert summary["buckets"]["0-30"]["kept"] == 2
    assert summary["dynamic_rules"] == ["3-10:score>=0.97,fill>=0.55,area>=10.0"]
