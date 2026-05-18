from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _install_pycocotools_stub() -> None:
    pycocotools = types.ModuleType("pycocotools")
    coco_mod = types.ModuleType("pycocotools.coco")
    cocoeval_mod = types.ModuleType("pycocotools.cocoeval")
    mask_mod = types.ModuleType("pycocotools.mask")

    class COCO:
        def __init__(self, ann_file: str | None = None) -> None:
            self.ann_file = ann_file

        def loadRes(self, rows):
            result = COCO(self.ann_file)
            result.rows = list(rows)
            return result

    class COCOeval:
        def __init__(self, coco_gt, coco_dt, iouType: str) -> None:
            self.iouType = iouType
            self.params = type("Params", (), {"maxDets": [1, 10, 100]})()
            self.stats = np.zeros(12, dtype=float)

        def evaluate(self) -> None:
            base = 0.1 if self.iouType == "bbox" else 0.2
            self.stats = np.array([base + self.params.maxDets[-1] / 1000.0] * 12, dtype=float)

        def accumulate(self) -> None:
            return None

        def summarize(self) -> None:
            return None

    def area(rle):
        return float(rle.get("area", 0.0))

    def toBbox(rle):
        return np.asarray(rle.get("bbox", [0.0, 0.0, 0.0, 0.0]), dtype=float)

    coco_mod.COCO = COCO
    cocoeval_mod.COCOeval = COCOeval
    mask_mod.area = area
    mask_mod.toBbox = toBbox
    pycocotools.coco = coco_mod
    pycocotools.cocoeval = cocoeval_mod
    pycocotools.mask = mask_mod
    sys.modules["pycocotools"] = pycocotools
    sys.modules["pycocotools.coco"] = coco_mod
    sys.modules["pycocotools.cocoeval"] = cocoeval_mod
    sys.modules["pycocotools.mask"] = mask_mod


_install_pycocotools_stub()

from tools import replay_official_coco_results as replay


def test_load_coco_payload_injects_metadata_without_rewriting_file(tmp_path: Path) -> None:
    ann_path = tmp_path / "ann.json"
    original = {
        "images": [{"id": 1, "file_name": "a.png", "height": 8, "width": 8}],
        "annotations": [],
        "categories": [{"id": 1, "name": "component"}],
    }
    ann_path.write_text(json.dumps(original) + "\n", encoding="utf-8")

    payload = replay.load_coco_payload(ann_path)

    assert payload["info"] == {}
    assert payload["licenses"] == []
    assert json.loads(ann_path.read_text(encoding="utf-8")) == original


def test_prepare_result_rows_drops_empty_masks_and_recomputes_bbox() -> None:
    valid_rle = {"size": [10, 12], "counts": "valid", "area": 12.0, "bbox": [2.0, 3.0, 3.0, 4.0]}
    empty_rle = {"size": [10, 12], "counts": "empty", "area": 0.0, "bbox": [0.0, 0.0, 0.0, 0.0]}

    prepared, summary = replay.prepare_result_rows(
        [
            {
                "image_id": 1,
                "category_id": 1,
                "score": 0.9,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "segmentation": valid_rle,
            },
            {
                "image_id": 1,
                "category_id": 1,
                "score": 0.1,
                "bbox": [1.0, 1.0, 2.0, 2.0],
                "segmentation": empty_rle,
            },
        ],
        drop_empty_masks=True,
        recompute_bbox=True,
    )

    assert summary == {
        "input": 2,
        "kept": 1,
        "dropped_empty_masks": 1,
        "recomputed_bboxes": 1,
    }
    assert prepared[0]["bbox"] == pytest.approx([2.0, 3.0, 3.0, 4.0])


def test_run_replay_writes_metric_shape_for_100_and_200_maxdets(tmp_path: Path) -> None:
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "height": 8, "width": 8}],
                "annotations": [],
                "categories": [{"id": 1, "name": "component"}],
            }
        ),
        encoding="utf-8",
    )
    results_path = tmp_path / "results.json"
    results_path.write_text("[]\n", encoding="utf-8")
    metrics_path = tmp_path / "metrics.cocoeval.json"

    metrics = replay.run_replay(
        ann_path=ann_path,
        results_path=results_path,
        output_metrics=metrics_path,
        max_dets=(100, 200),
    )

    assert metrics_path.is_file()
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metrics
    assert metrics["result_counts"] == {
        "input": 0,
        "kept": 0,
        "dropped_empty_masks": 0,
        "recomputed_bboxes": 0,
    }
    assert set(metrics["bbox"]) == {"maxDets100", "maxDets200"}
    assert set(metrics["segm"]) == {"maxDets100", "maxDets200"}
    assert metrics["bbox"]["maxDets200"]["AP"] == 0.0
    assert metrics["segm"]["maxDets100"]["AP"] == 0.0


def test_metric_block_uses_requested_maxdet_precision() -> None:
    class EvalObj:
        pass

    eval_obj = EvalObj()
    eval_obj.params = type(
        "Params",
        (),
        {
            "areaRngLbl": ["all", "small", "medium", "large"],
            "maxDets": [1, 10, 200],
            "iouThrs": np.array([0.50, 0.75]),
        },
    )()
    precision = np.full((2, 1, 1, 4, 3), -1.0, dtype=float)
    precision[:, :, :, 0, 2] = np.array([0.25, 0.75]).reshape(2, 1, 1)
    precision[:, :, :, 1, 2] = np.array([0.40, 0.60]).reshape(2, 1, 1)
    eval_obj.eval = {"precision": precision}

    metrics = replay._metric_block(eval_obj)

    assert metrics["AP"] == pytest.approx(50.0)
    assert metrics["AP50"] == pytest.approx(25.0)
    assert metrics["AP75"] == pytest.approx(75.0)
    assert metrics["APs"] == pytest.approx(50.0)
    assert metrics["APm"] == -100.0
