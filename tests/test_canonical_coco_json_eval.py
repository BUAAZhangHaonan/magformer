from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from magformer.engine.coco_json_eval import (
    build_coco_eval_contract,
    evaluate_standard_coco_rows,
    extract_coco_metrics,
    validate_standard_coco_rows,
)
from magformer.engine.evaluator import COCOEvaluator
from tools.evaluate_coco_json import evaluate_coco_json


def _rle(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def _mask_annotation(
    *,
    annotation_id: int,
    image_id: int,
    mask: np.ndarray,
) -> dict:
    segmentation = _rle(mask)
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": mask_utils.toBbox(segmentation).tolist(),
        "area": float(mask_utils.area(segmentation)),
        "iscrowd": 0,
        "segmentation": segmentation,
    }


def _build_coco(images: list[dict], annotations: list[dict]) -> COCO:
    coco = COCO()
    coco.dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }
    coco.createIndex()
    return coco


def _standard_mask_row(image_id: int, mask: np.ndarray, score: float = 1.0) -> dict:
    segmentation = _rle(mask)
    return {
        "image_id": image_id,
        "category_id": 1,
        "score": score,
        "bbox": mask_utils.toBbox(segmentation).tolist(),
        "segmentation": segmentation,
    }


def test_standard_xywh_is_preserved_and_matches_official_coco_metrics() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[3:8, 2:6] = 1
    coco_gt = _build_coco(
        [{"id": 1, "file_name": "one.png", "height": 16, "width": 16}],
        [_mask_annotation(annotation_id=1, image_id=1, mask=mask)],
    )
    row = _standard_mask_row(1, mask)
    contract = build_coco_eval_contract(coco_gt)

    canonical = validate_standard_coco_rows(
        coco_gt,
        [row],
        contract=contract,
        iou_types=("bbox", "segm"),
    )
    assert canonical[0]["bbox"] == [2.0, 3.0, 4.0, 5.0]

    metrics = evaluate_standard_coco_rows(
        coco_gt,
        [row],
        contract=contract,
        iou_types=("bbox", "segm"),
    )
    for iou_type in ("bbox", "segm"):
        rows_for_type = [copy.deepcopy(row)]
        if iou_type == "segm":
            rows_for_type[0].pop("bbox")
        official = COCOeval(coco_gt, coco_gt.loadRes(rows_for_type), iouType=iou_type)
        contract.apply(official)
        official.evaluate()
        official.accumulate()
        official_metrics = extract_coco_metrics(official, iou_type, max_dets=100)
        for key, expected in official_metrics.items():
            assert metrics[key] == pytest.approx(expected, abs=1.0e-12)

        assert metrics[f"{iou_type}_AP"] == pytest.approx(1.0, abs=1.0e-12)
        assert metrics[f"{iou_type}_AP80"] == pytest.approx(1.0, abs=1.0e-12)
        assert metrics[f"{iou_type}_AP95"] == pytest.approx(1.0, abs=1.0e-12)
        assert metrics[f"{iou_type}_AP_H"] == pytest.approx(1.0, abs=1.0e-12)
        assert metrics[f"{iou_type}_AR100"] == pytest.approx(1.0, abs=1.0e-12)


def test_nonempty_rle_with_zero_bbox_is_rejected_instead_of_repaired() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:10, 5:12] = 1
    coco_gt = _build_coco(
        [{"id": 1, "file_name": "one.png", "height": 16, "width": 16}],
        [_mask_annotation(annotation_id=1, image_id=1, mask=mask)],
    )
    row = _standard_mask_row(1, mask)
    row["bbox"] = [0.0, 0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="does not match RLE toBbox"):
        evaluate_standard_coco_rows(
            coco_gt,
            [row],
            contract=build_coco_eval_contract(coco_gt),
            iou_types=("bbox", "segm"),
        )


def test_online_evaluator_rejects_standard_xywh_rows_instead_of_double_converting() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[3:9, 4:11] = 1
    coco_gt = _build_coco(
        [{"id": 1, "file_name": "one.png", "height": 16, "width": 16}],
        [_mask_annotation(annotation_id=1, image_id=1, mask=mask)],
    )
    standard_row = _standard_mask_row(1, mask)
    assert standard_row["bbox"] == [4.0, 3.0, 7.0, 6.0]
    evaluator = COCOEvaluator(coco_gt=coco_gt, iou_types=["bbox", "segm"])
    evaluator.update([standard_row], image_ids=[1])

    with pytest.raises(ValueError, match="Standard COCO rows with XYWH"):
        evaluator.to_coco_results()


def test_segm_small_area_uses_mask_area_not_bbox_area() -> None:
    gt_mask = np.zeros((128, 128), dtype=np.uint8)
    gt_mask[10:20, 10:20] = 1
    false_mask = np.zeros((128, 128), dtype=np.uint8)
    false_mask[0, 0] = 1
    false_mask[99, 99] = 1
    coco_gt = _build_coco(
        [{"id": 1, "file_name": "one.png", "height": 128, "width": 128}],
        [_mask_annotation(annotation_id=1, image_id=1, mask=gt_mask)],
    )
    rows = [
        _standard_mask_row(1, false_mask, score=0.99),
        _standard_mask_row(1, gt_mask, score=0.90),
    ]
    contract = build_coco_eval_contract(coco_gt)

    metrics = evaluate_standard_coco_rows(
        coco_gt,
        rows,
        contract=contract,
        iou_types=("segm",),
    )

    buggy_dt = coco_gt.loadRes(rows)
    buggy_eval = COCOeval(coco_gt, buggy_dt, iouType="segm")
    contract.apply(buggy_eval)
    buggy_eval.evaluate()
    buggy_eval.accumulate()
    buggy_metrics = extract_coco_metrics(buggy_eval, "segm", max_dets=100)

    assert metrics["segm_APs"] < buggy_metrics["segm_APs"]
    assert buggy_dt.anns[1]["area"] == pytest.approx(10000.0)
    assert float(mask_utils.area(rows[0]["segmentation"])) == pytest.approx(2.0)


def test_full_gt_image_ids_include_images_without_predictions() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    coco_gt = _build_coco(
        [
            {"id": 1, "file_name": "one.png", "height": 16, "width": 16},
            {"id": 2, "file_name": "two.png", "height": 16, "width": 16},
        ],
        [
            _mask_annotation(annotation_id=1, image_id=1, mask=mask),
            _mask_annotation(annotation_id=2, image_id=2, mask=mask),
        ],
    )
    row = _standard_mask_row(1, mask)

    full_metrics = evaluate_standard_coco_rows(
        coco_gt,
        [row],
        contract=build_coco_eval_contract(coco_gt),
        iou_types=("segm",),
    )
    subset_metrics = evaluate_standard_coco_rows(
        coco_gt,
        [row],
        contract=build_coco_eval_contract(coco_gt, image_ids=[1]),
        iou_types=("segm",),
    )

    assert 0.0 < full_metrics["segm_AP"] < subset_metrics["segm_AP"]
    assert subset_metrics["segm_AP"] == pytest.approx(1.0, abs=1.0e-12)


def test_more_than_max_dets_is_rejected_before_cocoeval() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    coco_gt = _build_coco(
        [{"id": 1, "file_name": "one.png", "height": 16, "width": 16}],
        [_mask_annotation(annotation_id=1, image_id=1, mask=mask)],
    )
    row = _standard_mask_row(1, mask)

    with pytest.raises(ValueError, match="exceed maxDets=100"):
        evaluate_standard_coco_rows(
            coco_gt,
            [copy.deepcopy(row) for _ in range(101)],
            contract=build_coco_eval_contract(coco_gt),
            iou_types=("bbox", "segm"),
        )


def test_online_and_standard_json_paths_have_identical_metrics() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[3:9, 4:11] = 1
    segmentation = _rle(mask)
    coco_gt = _build_coco(
        [{"id": 1, "file_name": "one.png", "height": 16, "width": 16}],
        [_mask_annotation(annotation_id=1, image_id=1, mask=mask)],
    )
    online = COCOEvaluator(coco_gt=coco_gt, iou_types=["bbox", "segm"], max_dets=100)
    online.update(
        [
            {
                "image_id": 1,
                "category_id": 1,
                "score": 1.0,
                "bbox": [4.0, 3.0, 11.0, 9.0],
                "mask": segmentation,
            }
        ],
        image_ids=[1],
    )

    online_metrics = online.summarize()
    json_metrics = evaluate_standard_coco_rows(
        coco_gt,
        online.to_coco_results(),
        contract=build_coco_eval_contract(coco_gt),
        iou_types=("bbox", "segm"),
    )

    assert online_metrics == json_metrics


def test_offline_cli_function_records_exact_contract(tmp_path: Path) -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[3:9, 4:11] = 1
    annotation_path = tmp_path / "instances_val.json"
    results_path = tmp_path / "coco_instances_results.json"
    coco_gt = _build_coco(
        [{"id": 1, "file_name": "one.png", "height": 16, "width": 16}],
        [_mask_annotation(annotation_id=1, image_id=1, mask=mask)],
    )
    annotation_path.write_text(json.dumps(coco_gt.dataset), encoding="utf-8")
    results_path.write_text(json.dumps([_standard_mask_row(1, mask)]), encoding="utf-8")

    payload = evaluate_coco_json(
        annotation_path=annotation_path,
        results_path=results_path,
        expected_image_count=1,
    )

    assert payload["metric_scale"] == "fraction"
    assert payload["num_images"] == 1
    assert payload["num_predictions"] == 1
    assert payload["contract"]["max_dets"] == [1, 10, 100]
    assert payload["metrics"]["segm_AP_H"] == pytest.approx(1.0)
