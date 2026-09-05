from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval as PyCOCOeval
from torch.utils.data import DataLoader, Dataset

from magformer.engine import evaluator as evaluator_module
from magformer.engine.eval_runtime import (
    _build_prefix_limited_eval_loader,
    _rank_stride_eval_indices,
    _raise_global_contract_error_on_all_ranks,
    _select_global_eval_indices,
)
from magformer.engine.evaluator import COCOEvaluator


def _prediction(image_id: int) -> dict:
    return {
        "image_id": image_id,
        "category_id": 1,
        "score": 0.9,
        "bbox": [1.0, 1.0, 3.0, 3.0],
    }


def test_global_eval_indices_are_deterministic_disjoint_and_exact() -> None:
    selected = _select_global_eval_indices(10, 7, seed=42)
    assert selected == _select_global_eval_indices(10, 7, seed=42)
    assert len(selected) == len(set(selected)) == 7

    shards = [
        _rank_stride_eval_indices(selected, rank=rank, world_size=3)
        for rank in range(3)
    ]
    assert set(shards[0]).isdisjoint(shards[1])
    assert set(shards[0]).isdisjoint(shards[2])
    assert set(shards[1]).isdisjoint(shards[2])
    assert set().union(*map(set, shards)) == set(selected)
    assert _select_global_eval_indices(4, 999) == [0, 1, 2, 3]


def test_coco_evaluator_rejects_missing_unknown_and_duplicate_image_ids() -> None:
    evaluator = COCOEvaluator(coco_gt=SimpleNamespace(), iou_types=["bbox"])
    with pytest.raises(ValueError, match="at least one image_id"):
        evaluator.update([], image_ids=[])
    with pytest.raises(ValueError, match="outside the evaluated batch"):
        evaluator.update([_prediction(2)], image_ids=[1])

    evaluator.update([], image_ids=[1])
    with pytest.raises(ValueError, match="more than once"):
        evaluator.update([], image_ids=[1])


def test_coco_evaluator_gathers_results_and_all_evaluated_ids(monkeypatch) -> None:
    evaluator = COCOEvaluator(coco_gt=SimpleNamespace(), iou_types=["bbox"])
    first = _prediction(1)
    second = _prediction(2)
    evaluator.update([first], image_ids=[1])
    gathered_values = iter(([[first], [second]], [[1], [2]]))

    def fake_all_gather_object(output, value) -> None:
        del value
        values = next(gathered_values)
        output[:] = values

    monkeypatch.setattr(evaluator_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(evaluator_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(evaluator_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        evaluator_module.dist, "all_gather_object", fake_all_gather_object
    )

    evaluator.synchronize_between_processes()

    assert evaluator.image_ids == [1, 2]
    assert [row["image_id"] for row in evaluator.results] == [1, 2]


class _FakeCOCO:
    def __init__(self) -> None:
        self.imgs = {
            1: {"id": 1, "height": 16, "width": 16},
            2: {"id": 2, "height": 16, "width": 16},
        }
        self.dataset = {
            "images": list(self.imgs.values()),
            "categories": [{"id": 1, "name": "object"}],
        }

    def getImgIds(self):
        return list(self.imgs)

    def getCatIds(self):
        return [1]

    def loadRes(self, rows):
        return list(rows)


class _CapturingCOCOeval:
    instances = []

    def __init__(self, coco_gt, coco_dt, iouType="bbox") -> None:
        del coco_gt, coco_dt
        self.iouType = iouType
        self.params = SimpleNamespace(
            imgIds=[],
            catIds=[1],
            maxDets=[1, 10, 100],
            iouThrs=np.arange(0.5, 1.0, 0.05),
            recThrs=np.linspace(0.0, 1.0, 101),
            areaRng=[[0.0, 1.0e10], [0.0, 1024.0], [1024.0, 9216.0], [9216.0, 1.0e10]],
            areaRngLbl=["all", "small", "medium", "large"],
            useCats=1,
        )
        self.stats = np.ones(12, dtype=np.float64)
        self.eval = {}
        self.__class__.instances.append(self)

    def evaluate(self) -> None:
        return None

    def accumulate(self) -> None:
        self.eval["precision"] = np.ones(
            (len(self.params.iouThrs), 101, 1, 4, len(self.params.maxDets)), dtype=np.float64
        )
        self.eval["recall"] = np.ones(
            (len(self.params.iouThrs), 1, 4, len(self.params.maxDets)), dtype=np.float64
        )

    def summarize(self) -> None:
        return None


def test_coco_evaluator_includes_evaluated_images_without_predictions(
    monkeypatch,
) -> None:
    _CapturingCOCOeval.instances.clear()
    monkeypatch.setattr(evaluator_module, "COCOeval", _CapturingCOCOeval)
    evaluator = COCOEvaluator(coco_gt=_FakeCOCO(), iou_types=["bbox"])
    evaluator.update([_prediction(1)], image_ids=[1, 2])

    evaluator.summarize()

    assert _CapturingCOCOeval.instances[-1].params.imgIds == [1, 2]


def test_coco_evaluator_uses_half_open_mask_boxes_for_dense_and_rle_masks() -> None:
    mask = np.zeros((8, 9), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")

    evaluator = COCOEvaluator(coco_gt=SimpleNamespace(), iou_types=["bbox"])

    assert evaluator._bbox_from_mask(mask) == [3.0, 2.0, 7.0, 5.0]
    assert evaluator._bbox_from_mask(rle) == [3.0, 2.0, 7.0, 5.0]

    evaluator.update(
        [{"image_id": 1, "category_id": 1, "score": 0.9, "mask": rle}],
        image_ids=[1],
    )
    assert evaluator.to_coco_results()[0]["bbox"] == [3.0, 2.0, 4.0, 3.0]


def test_coco_evaluator_applies_custom_iou_thresholds(monkeypatch) -> None:
    _CapturingCOCOeval.instances.clear()
    monkeypatch.setattr(evaluator_module, "COCOeval", _CapturingCOCOeval)
    thresholds = [0.50, 0.75, 0.80, 0.85, 0.90, 0.95]
    evaluator = COCOEvaluator(
        coco_gt=_FakeCOCO(),
        iou_types=["bbox"],
        iou_thresholds=thresholds,
    )
    evaluator.update([_prediction(1)], image_ids=[1])

    evaluator.summarize()

    np.testing.assert_array_equal(
        _CapturingCOCOeval.instances[-1].params.iouThrs,
        np.asarray(thresholds),
    )


@pytest.mark.parametrize("max_dets", [-1, 0, 10])
def test_coco_evaluator_rejects_max_dets_that_duplicate_standard_slices(
    max_dets: int,
) -> None:
    with pytest.raises(ValueError, match="greater than 10"):
        COCOEvaluator(
            coco_gt=SimpleNamespace(), iou_types=["bbox"], max_dets=max_dets
        )


def _single_box_coco() -> COCO:
    coco = COCO()
    coco.dataset = {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "file_name": "one.png", "width": 16, "height": 16}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2.0, 3.0, 4.0, 5.0],
                "area": 20.0,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "object"}],
    }
    coco.createIndex()
    return coco


def _perfect_box_prediction() -> dict:
    return {
        "image_id": 1,
        "category_id": 1,
        "score": 1.0,
        "bbox": [2.0, 3.0, 6.0, 8.0],
    }


def test_tensor_metrics_at_max_dets_100_match_standard_coco_stats() -> None:
    coco_gt = _single_box_coco()
    prediction = _perfect_box_prediction()
    evaluator = COCOEvaluator(coco_gt=coco_gt, iou_types=["bbox"], max_dets=100)
    evaluator.update([prediction], image_ids=[1])

    metrics = evaluator.summarize()

    coco_dt = coco_gt.loadRes(evaluator.to_coco_results())
    standard_eval = PyCOCOeval(coco_gt, coco_dt, iouType="bbox")
    standard_eval.params.imgIds = [1]
    standard_eval.evaluate()
    standard_eval.accumulate()
    standard_eval.summarize()
    stats_mapping = {
        "bbox_AP": 0,
        "bbox_AP50": 1,
        "bbox_AP75": 2,
        "bbox_APs": 3,
        "bbox_APm": 4,
        "bbox_APl": 5,
        "bbox_AR1": 6,
        "bbox_AR10": 7,
        "bbox_AR100": 8,
        "bbox_ARs": 9,
        "bbox_ARm": 10,
        "bbox_ARl": 11,
    }
    for metric_name, stats_index in stats_mapping.items():
        assert metrics[metric_name] == pytest.approx(
            standard_eval.stats[stats_index], abs=1e-12
        )


def test_perfect_prediction_at_max_dets_300_has_perfect_ap() -> None:
    evaluator = COCOEvaluator(
        coco_gt=_single_box_coco(), iou_types=["bbox"], max_dets=300
    )
    evaluator.update([_perfect_box_prediction()], image_ids=[1])

    metrics = evaluator.summarize()

    assert metrics["bbox_AP"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["bbox_AP50"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["bbox_AP75"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["bbox_AR300"] == pytest.approx(1.0, abs=1e-12)
    assert "bbox_AR100" not in metrics


class _ImageIdDataset(Dataset):
    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> dict:
        return {"image_ids": 100 + index}


def test_prefix_limited_loader_is_exact_with_batches_larger_than_one() -> None:
    loader = DataLoader(_ImageIdDataset(), batch_size=2, shuffle=False)

    limited_loader, target_images = _build_prefix_limited_eval_loader(loader, 3)
    observed_ids = [
        int(image_id)
        for batch in limited_loader
        for image_id in batch["image_ids"]
    ]

    assert target_images == 3
    assert observed_ids == [100, 101, 102]
    with pytest.raises(ValueError, match="max_images must be positive"):
        _build_prefix_limited_eval_loader(loader, 0)


def test_rank_zero_contract_error_is_broadcast_before_all_ranks_raise(
    monkeypatch,
) -> None:
    message = (
        "Evaluation image count does not match the global index plan: "
        "expected=3, observed=4"
    )
    monkeypatch.setattr(
        "magformer.engine.eval_runtime.dist.is_available", lambda: True
    )
    monkeypatch.setattr(
        "magformer.engine.eval_runtime.dist.is_initialized", lambda: True
    )

    root_payloads = []

    def root_broadcast(payload, src) -> None:
        assert src == 0
        root_payloads.append(payload[0])

    monkeypatch.setattr(
        "magformer.engine.eval_runtime.dist.broadcast_object_list", root_broadcast
    )
    with pytest.raises(RuntimeError, match="expected=3, observed=4"):
        _raise_global_contract_error_on_all_ranks(message, rank=0)
    assert root_payloads == [message]

    def worker_broadcast(payload, src) -> None:
        assert src == 0
        assert payload == [None]
        payload[0] = message

    monkeypatch.setattr(
        "magformer.engine.eval_runtime.dist.broadcast_object_list", worker_broadcast
    )
    with pytest.raises(RuntimeError, match="expected=3, observed=4"):
        _raise_global_contract_error_on_all_ranks(None, rank=1)


def test_specialized_eval_entrypoints_default_to_zero_score_thresholds(
    monkeypatch,
) -> None:
    from tools import eval_hierarchical_fusion, evaluate_sahi, evaluate_tta

    monkeypatch.setattr(
        sys, "argv", ["evaluate_tta.py", "--config", "x", "--checkpoint", "x"]
    )
    tta_args = evaluate_tta.parse_args()
    assert tta_args.score_thresh == 0.0
    assert tta_args.export_score_thresh == 0.0

    monkeypatch.setattr(
        sys, "argv", ["evaluate_sahi.py", "--config", "x", "--checkpoint", "x"]
    )
    sahi_args = evaluate_sahi.parse_args()
    assert sahi_args.score_thresh == 0.0
    assert sahi_args.export_score_thresh == 0.0

    monkeypatch.setattr(sys, "argv", ["eval_hierarchical_fusion.py", "--config", "x"])
    hierarchical_args = eval_hierarchical_fusion.parse_args()
    assert hierarchical_args.score_threshold == 0.0
    assert hierarchical_args.export_score_threshold == 0.0


def test_specialized_eval_entrypoints_allow_zero_exported_predictions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for relative_path in ("tools/evaluate_tta.py", "tools/evaluate_sahi.py"):
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "exported zero COCO predictions" not in source

    evaluator = COCOEvaluator(coco_gt=_single_box_coco(), iou_types=["bbox"])
    evaluator.update([], image_ids=[1])
    metrics = evaluator.summarize()
    assert metrics["bbox_AP"] == 0.0
    assert metrics["bbox_AP50"] == 0.0
    assert metrics["bbox_AR100"] == 0.0


def test_high_iou_metrics_use_valid_precision_values_only() -> None:
    precision = np.full((10, 2, 1, 4, 3), -1.0, dtype=np.float64)
    precision[6, :, 0, 0, 2] = [0.8, -1.0]
    precision[7, :, 0, 0, 2] = [0.6, 0.4]
    precision[8, :, 0, 0, 2] = [0.3, 0.1]
    precision[9, :, 0, 0, 2] = [-1.0, 0.1]
    coco_eval = SimpleNamespace(
        stats=np.zeros(12, dtype=np.float64),
        params=SimpleNamespace(
            iouThrs=np.arange(0.5, 1.0, 0.05),
            recThrs=np.linspace(0.0, 1.0, 2),
            catIds=[1],
            useCats=1,
            areaRng=[[0.0, 1.0e10], [0.0, 1024.0], [1024.0, 9216.0], [9216.0, 1.0e10]],
            areaRngLbl=["all", "small", "medium", "large"],
            maxDets=[1, 10, 100],
        ),
        eval={"precision": precision, "recall": np.full((10, 1, 4, 3), -1.0)},
    )
    evaluator = COCOEvaluator(
        coco_gt=SimpleNamespace(), iou_types=["bbox"], max_dets=100
    )

    metrics = evaluator._extract_metrics(coco_eval, "bbox")

    assert metrics["bbox_AP80"] == pytest.approx(0.8)
    assert metrics["bbox_AP85"] == pytest.approx(0.5)
    assert metrics["bbox_AP90"] == pytest.approx(0.2)
    assert metrics["bbox_AP95"] == pytest.approx(0.1)
    assert metrics["bbox_AP_H"] == pytest.approx((0.8 + 0.5 + 0.2 + 0.1) / 4.0)


def test_mask2former_instance_boxes_are_derived_from_predicted_masks() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mask2former_root = repo_root / "baselines" / "Mask2Former"
    detectron2_root = repo_root / "baselines" / "detectron2"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(mask2former_root), str(detectron2_root), env.get("PYTHONPATH", "")]
    )
    code = """
import json
from types import SimpleNamespace
import torch
from mask2former.maskformer_model import MaskFormer

model_contract = SimpleNamespace(
    sem_seg_head=SimpleNamespace(num_classes=1),
    device=torch.device('cpu'),
    num_queries=1,
    test_topk_per_image=1,
    panoptic_on=False,
)
mask_cls = torch.tensor([[10.0, -10.0]])
mask_pred = torch.full((1, 8, 9), -10.0)
mask_pred[0, 2:6, 3:8] = 10.0
result = MaskFormer.instance_inference(model_contract, mask_cls, mask_pred)
print(json.dumps({
    'boxes': result.pred_boxes.tensor.tolist(),
    'masks_match': bool(torch.equal(result.pred_masks.bool(), mask_pred > 0)),
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["boxes"] == [[3.0, 2.0, 8.0, 6.0]]
    assert payload["masks_match"] is True
