from __future__ import annotations

import json
import sys
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _ensure_pycocotools_stub() -> None:
    if "pycocotools" in sys.modules:
        return

    pycocotools = types.ModuleType("pycocotools")
    coco_mod = types.ModuleType("pycocotools.coco")
    cocoeval_mod = types.ModuleType("pycocotools.cocoeval")
    mask_mod = types.ModuleType("pycocotools.mask")

    class COCO:
        def __init__(self, ann_file=None):
            self.ann_file = ann_file
            self.dataset = {}
            self.results = []
            if ann_file is not None:
                with open(ann_file, "r", encoding="utf-8") as handle:
                    self.dataset = json.load(handle)

        def loadRes(self, coco_results):
            result = COCO(None)
            result.dataset = self.dataset
            result.results = list(coco_results)
            return result

    class COCOeval:
        def __init__(self, coco_gt, coco_dt, iouType="bbox"):
            self.cocoGt = coco_gt
            self.cocoDt = coco_dt
            self.iouType = iouType
            self.params = types.SimpleNamespace(maxDets=[1, 10, 100], imgIds=[])
            self.stats = np.zeros(12, dtype=float)

        def evaluate(self):
            def _bbox_iou(box_a, box_b):
                ax, ay, aw, ah = [float(value) for value in box_a]
                bx, by, bw, bh = [float(value) for value in box_b]
                a_x2 = ax + aw
                a_y2 = ay + ah
                b_x2 = bx + bw
                b_y2 = by + bh
                inter_w = max(0.0, min(a_x2, b_x2) - max(ax, bx))
                inter_h = max(0.0, min(a_y2, b_y2) - max(ay, by))
                inter = inter_w * inter_h
                union = max(aw * ah + bw * bh - inter, 1e-12)
                return inter / union

            img_ids = {int(v) for v in getattr(self.params, "imgIds", [])}
            annotations = [
                ann
                for ann in self.cocoGt.dataset.get("annotations", [])
                if not img_ids or int(ann.get("image_id", -1)) in img_ids
            ]
            detections = [
                det
                for det in getattr(self.cocoDt, "results", [])
                if not img_ids or int(det.get("image_id", -1)) in img_ids
            ]
            value = 0.0
            if annotations and detections:
                gt_bbox = annotations[0].get("bbox", [0.0, 0.0, 0.0, 0.0])
                dt_bbox = detections[0].get("bbox", [0.0, 0.0, 0.0, 0.0])
                value = 1.0 if _bbox_iou(gt_bbox, dt_bbox) >= 0.5 else 0.0
            self.stats = np.full(12, value, dtype=float)

        def accumulate(self):
            return None

        def summarize(self):
            return None

    def encode(array):
        ys, xs = np.where(array > 0)
        bbox = [0.0, 0.0, 0.0, 0.0]
        if len(xs) > 0 and len(ys) > 0:
            bbox = [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() - xs.min() + 1),
                float(ys.max() - ys.min() + 1),
            ]
        return {
            "size": list(array.shape),
            "counts": b"1",
            "bbox": bbox,
        }

    def toBbox(rle):
        return np.asarray(rle.get("bbox", [0.0, 0.0, 0.0, 0.0]), dtype=float)

    coco_mod.COCO = COCO
    cocoeval_mod.COCOeval = COCOeval
    mask_mod.encode = encode
    mask_mod.toBbox = toBbox

    pycocotools.coco = coco_mod
    pycocotools.cocoeval = cocoeval_mod
    pycocotools.mask = mask_mod

    sys.modules["pycocotools"] = pycocotools
    sys.modules["pycocotools.coco"] = coco_mod
    sys.modules["pycocotools.cocoeval"] = cocoeval_mod
    sys.modules["pycocotools.mask"] = mask_mod


_ensure_pycocotools_stub()

from magformer.engine.trainer import DDPTrainer, Trainer
from pycocotools.coco import COCO


def _write_min_coco_dataset(root: Path) -> COCO:
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)

    (root / "images" / "val" / "0001.png").write_bytes(b"placeholder-image")

    ann = {
        "images": [{"id": 1, "file_name": "0001.png", "width": 32, "height": 32}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 8, 16, 16],
                "area": 256,
                "iscrowd": 0,
                "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]],
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    ann_path = root / "annotations" / "instances_val.json"
    ann_path.write_text(json.dumps(ann) + "\n", encoding="utf-8")
    return COCO(str(ann_path))


def _write_two_image_coco_dataset(root: Path) -> COCO:
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    for idx in (1, 2):
        (root / "images" / "val" / f"{idx:04d}.png").write_bytes(b"placeholder-image")

    ann = {
        "images": [
            {"id": 1, "file_name": "0001.png", "width": 32, "height": 32},
            {"id": 2, "file_name": "0002.png", "width": 32, "height": 32},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 8, 16, 16],
                "area": 256,
                "iscrowd": 0,
                "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]],
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [0, 0, 6, 6],
                "area": 36,
                "iscrowd": 0,
                "segmentation": [[0, 0, 6, 0, 6, 6, 0, 6]],
            },
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    ann_path = root / "annotations" / "instances_val.json"
    ann_path.write_text(json.dumps(ann) + "\n", encoding="utf-8")
    return COCO(str(ann_path))


class _EvalOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *args, **kwargs):
        raise AssertionError("evaluation should use forward_inference_raw, not forward()")

    def forward_inference_raw(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        padding_masks=None,
        depth_noise_masks=None,
    ):
        del depths, padding_masks, depth_noise_masks
        batch_size = int(images.shape[0])
        masks = images.new_zeros((1, 32, 32))
        masks[:, 8:24, 8:24] = 1.0
        predictions = []
        for _ in range(batch_size):
            predictions.append(
                {
                    "scores": torch.tensor([0.99], dtype=images.dtype, device=images.device),
                    "category_ids": torch.tensor([0], dtype=torch.long, device=images.device),
                    "masks": masks.clone(),
                }
            )
        return {"predictions": predictions}


class _WrongEvalModel(_EvalOnlyModel):
    def forward_inference_raw(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        padding_masks=None,
        depth_noise_masks=None,
    ):
        del depths, padding_masks, depth_noise_masks
        batch_size = int(images.shape[0])
        masks = images.new_zeros((1, 32, 32))
        masks[:, :8, :8] = 1.0
        predictions = []
        for _ in range(batch_size):
            predictions.append(
                {
                    "scores": torch.tensor([0.99], dtype=images.dtype, device=images.device),
                    "category_ids": torch.tensor([0], dtype=torch.long, device=images.device),
                    "masks": masks.clone(),
                }
            )
        return {"predictions": predictions}


class _EmptyEvalModel(_EvalOnlyModel):
    def forward_inference_raw(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        padding_masks=None,
        depth_noise_masks=None,
    ):
        del depths, padding_masks, depth_noise_masks
        batch_size = int(images.shape[0])
        predictions = []
        for _ in range(batch_size):
            predictions.append(
                {
                    "scores": torch.zeros((0,), dtype=images.dtype, device=images.device),
                    "category_ids": torch.zeros((0,), dtype=torch.long, device=images.device),
                    "masks": images.new_zeros((0, 32, 32)),
                }
            )
        return {"predictions": predictions}



def test_run_inference_evaluation_max_images_counts_images_not_batches(tmp_path: Path) -> None:
    coco = _write_two_image_coco_dataset(tmp_path / "ds")
    loader = [
        {
            "images": torch.zeros(2, 3, 32, 32),
            "depths": torch.zeros(2, 1, 32, 32),
            "image_ids": torch.tensor([1, 2]),
        }
    ]

    from magformer.engine.eval_runtime import run_inference_evaluation

    result = run_inference_evaluation(
        _EvalOnlyModel(),
        loader,
        coco_gt=coco,
        device=torch.device("cpu"),
        output_dir=tmp_path / "out",
        amp_enabled=False,
        iou_types=["bbox"],
        max_images=1,
    )

    assert result.log_dict["val/diag_num_eval_images"] == 1.0
    assert result.log_dict["val/diag_num_predictions"] == 1.0


def test_subset_evaluation_sets_cocoeval_img_ids(tmp_path: Path) -> None:
    coco = _write_two_image_coco_dataset(tmp_path / "ds")
    loader = [
        {
            "images": torch.zeros(1, 3, 32, 32),
            "depths": torch.zeros(1, 1, 32, 32),
            "image_ids": torch.tensor([1]),
        }
    ]

    from magformer.engine.eval_runtime import run_inference_evaluation

    result = run_inference_evaluation(
        _EvalOnlyModel(),
        loader,
        coco_gt=coco,
        device=torch.device("cpu"),
        output_dir=tmp_path / "out",
        amp_enabled=False,
        iou_types=["bbox"],
        max_images=1,
    )

    assert result.coco_metrics["bbox_AP"] == pytest.approx(1.0)


def test_cocoevaluator_forwards_deduped_img_ids_to_cocoeval(monkeypatch, tmp_path: Path) -> None:
    coco = _write_two_image_coco_dataset(tmp_path / "ds")

    import magformer.engine.evaluator as evaluator_mod
    from magformer.engine.evaluator import COCOEvaluator

    captured = {}

    class _CapturingCOCOeval:
        def __init__(self, coco_gt, coco_dt, iouType="bbox"):
            del coco_gt, coco_dt
            captured["iou_type"] = iouType
            self.params = SimpleNamespace(maxDets=[1, 10, 100], imgIds=[])
            self.stats = np.ones(12, dtype=float)

        def evaluate(self):
            captured["img_ids"] = list(self.params.imgIds)

        def accumulate(self):
            return None

        def summarize(self):
            return None

    monkeypatch.setattr(evaluator_mod, "COCOeval", _CapturingCOCOeval)
    evaluator = COCOEvaluator(coco_gt=coco, iou_types=["bbox"], image_ids=[2, 2, 1, 2])
    evaluator.update(
        [
            {"image_id": 2, "category_id": 1, "score": 0.9, "bbox": [0.0, 0.0, 6.0, 6.0]},
            {"image_id": 1, "category_id": 1, "score": 0.8, "bbox": [8.0, 8.0, 24.0, 24.0]},
        ]
    )

    evaluator.summarize()

    assert captured["iou_type"] == "bbox"
    assert captured["img_ids"] == [2, 1]


def test_bbox_only_evaluation_does_not_emit_segm_metrics_or_segmentation_results(tmp_path: Path) -> None:
    coco = _write_min_coco_dataset(tmp_path / "ds")
    loader = [
        {
            "images": torch.zeros(1, 3, 32, 32),
            "depths": torch.zeros(1, 1, 32, 32),
            "image_ids": torch.tensor([1]),
        }
    ]

    from magformer.engine.eval_runtime import run_inference_evaluation

    result = run_inference_evaluation(
        _EvalOnlyModel(),
        loader,
        coco_gt=coco,
        device=torch.device("cpu"),
        output_dir=tmp_path / "out",
        amp_enabled=False,
        iou_types=["bbox"],
        max_images=1,
    )

    assert all(not key.startswith("segm_") for key in result.coco_metrics)
    assert "val/segm_AP" not in result.log_dict
    rows = json.loads(result.coco_results_path.read_text(encoding="utf-8"))
    assert rows
    assert all("segmentation" not in row for row in rows)


def test_duplicate_image_predictions_are_deduped_before_dump(tmp_path: Path) -> None:
    coco = _write_min_coco_dataset(tmp_path / "ds")
    loader = [
        {
            "images": torch.zeros(2, 3, 32, 32),
            "depths": torch.zeros(2, 1, 32, 32),
            "image_ids": torch.tensor([1, 1]),
        }
    ]

    from magformer.engine.eval_runtime import run_inference_evaluation

    result = run_inference_evaluation(
        _EvalOnlyModel(),
        loader,
        coco_gt=coco,
        device=torch.device("cpu"),
        output_dir=tmp_path / "out",
        amp_enabled=False,
        iou_types=["bbox"],
    )

    rows = json.loads(result.coco_results_path.read_text(encoding="utf-8"))
    assert result.log_dict["val/diag_num_eval_images"] == 1.0
    assert len(rows) == 1

def test_trainer_evaluate_uses_inference_contract_and_logs_metrics_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    coco = _write_min_coco_dataset(tmp_path / "ds")
    model = _EvalOnlyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        train_loader=[{"images": torch.zeros(1, 3, 32, 32), "depths": torch.zeros(1, 1, 32, 32)}],
        val_loader=[
            {
                "images": torch.zeros(1, 3, 32, 32),
                "depths": torch.zeros(1, 1, 32, 32),
                "image_ids": torch.tensor([1]),
            }
        ],
        val_dataset=SimpleNamespace(coco=coco),
        device=torch.device("cpu"),
        output_dir=str(tmp_path / "out"),
        max_iter=1,
        eval_period=1,
        checkpoint_period=100,
        log_period=1,
        amp_enabled=False,
    )
    trainer.current_iter = 7

    logged = {}
    checkpoint_calls = []

    monkeypatch.setattr(trainer, "_save_eval_visualization", lambda batch, outputs: None)
    trainer.logger = SimpleNamespace(
        log_scalars=lambda main_tag, tag_scalar_dict, step: logged.update(tag_scalar_dict),
        close=lambda: None,
    )
    monkeypatch.setattr(trainer, "save_checkpoint", lambda is_best=False: checkpoint_calls.append(bool(is_best)))

    metrics = trainer.evaluate()

    assert "val/segm_AP" in metrics
    assert "val/mAP" in metrics
    assert metrics["val/diag_num_eval_images"] == 1.0
    assert "val/loss" not in metrics
    assert "val/segm_AP" in logged
    assert "val/loss" not in logged
    assert trainer.best_metric == metrics["val/mAP"]
    assert checkpoint_calls == [True]


def test_trainer_eval_can_be_diagnostic_without_selecting_model_best(
    monkeypatch,
    tmp_path: Path,
) -> None:
    coco = _write_min_coco_dataset(tmp_path / "ds")
    model = _EvalOnlyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        train_loader=[],
        val_loader=[
            {
                "images": torch.zeros(1, 3, 32, 32),
                "depths": torch.zeros(1, 1, 32, 32),
                "image_ids": torch.tensor([1]),
            }
        ],
        val_dataset=SimpleNamespace(coco=coco),
        config={
            "runtime": {
                "eval_iou_types": ["bbox"],
                "eval_max_images": 1,
                "eval_saves_best": False,
            }
        },
        device=torch.device("cpu"),
        output_dir=str(tmp_path / "out"),
        max_iter=1,
        eval_period=1,
        checkpoint_period=100,
        log_period=1,
        amp_enabled=False,
    )
    trainer.current_iter = 7
    trainer.best_metric = 0.0

    logged = {}
    checkpoint_calls = []

    monkeypatch.setattr(trainer, "_save_eval_visualization", lambda batch, outputs: None)
    monkeypatch.setattr(trainer, "_console_log", lambda message: None)
    trainer.logger = SimpleNamespace(
        log_scalars=lambda main_tag, tag_scalar_dict, step: logged.update(tag_scalar_dict),
        close=lambda: None,
    )
    monkeypatch.setattr(trainer, "save_checkpoint", lambda is_best=False: checkpoint_calls.append(bool(is_best)))

    metrics = trainer.evaluate()

    assert "val/bbox_AP" in metrics
    assert logged["val/mAP"] == metrics["val/mAP"]
    assert trainer.best_metric == 0.0
    assert checkpoint_calls == []


class _DDPWrapper:
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module

    def eval(self):
        self.module.eval()
        return self

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self


def test_ddp_trainer_evaluate_logs_metrics_only_on_rank_zero(
    monkeypatch,
    tmp_path: Path,
) -> None:
    coco = _write_min_coco_dataset(tmp_path / "ds")
    model = _EvalOnlyModel()
    trainer = DDPTrainer.__new__(DDPTrainer)
    trainer.model = _DDPWrapper(model)
    trainer.val_loader = [
        {
            "images": torch.zeros(1, 3, 32, 32),
            "depths": torch.zeros(1, 1, 32, 32),
            "image_ids": [1],
        }
    ]
    trainer.val_dataset = SimpleNamespace(coco=coco, category_ids=[1])
    trainer.device = torch.device("cpu")
    trainer.output_dir = tmp_path / "out"
    trainer.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.visualization_dir = trainer.output_dir / "visualizations"
    trainer.visualization_dir.mkdir(parents=True, exist_ok=True)
    trainer.metrics_log_file = trainer.output_dir / "metrics_log.jsonl"
    trainer.metrics_csv_file = trainer.output_dir / "metrics_log.csv"
    trainer._csv_header_written = False
    trainer.rank = 0
    trainer.world_size = 2
    trainer.amp_enabled = False
    trainer.current_iter = 9
    trainer.best_metric = 0.0
    trainer.max_iter = 1
    trainer._iter_time_window_sec = deque(maxlen=20)
    trainer._train_start_monotonic = None

    logged = {}
    checkpoint_calls = []

    def _fake_all_gather_object(output, value):
        filler = [] if isinstance(value, list) else 0
        for idx in range(len(output)):
            output[idx] = filler
        output[0] = value

    monkeypatch.setattr(trainer, "_save_eval_visualization", lambda batch, outputs: None)
    monkeypatch.setattr(trainer, "_console_log", lambda message: None)
    trainer.logger = SimpleNamespace(
        log_scalars=lambda main_tag, tag_scalar_dict, step: logged.update(tag_scalar_dict),
        close=lambda: None,
    )
    monkeypatch.setattr(trainer, "save_checkpoint", lambda is_best=False: checkpoint_calls.append(bool(is_best)))

    monkeypatch.setattr("magformer.engine.eval_runtime.dist.is_available", lambda: True)
    monkeypatch.setattr("magformer.engine.eval_runtime.dist.is_initialized", lambda: True)
    monkeypatch.setattr("magformer.engine.eval_runtime.dist.get_rank", lambda: 0)
    monkeypatch.setattr("magformer.engine.eval_runtime.dist.get_world_size", lambda: 2)
    monkeypatch.setattr("magformer.engine.eval_runtime.dist.all_gather_object", _fake_all_gather_object)
    monkeypatch.setattr("magformer.engine.evaluator.dist.is_available", lambda: True)
    monkeypatch.setattr("magformer.engine.evaluator.dist.is_initialized", lambda: True)
    monkeypatch.setattr("magformer.engine.evaluator.dist.get_world_size", lambda: 2)
    monkeypatch.setattr("magformer.engine.evaluator.dist.all_gather_object", _fake_all_gather_object)
    monkeypatch.setattr("magformer.engine.trainer.dist.is_available", lambda: True)
    monkeypatch.setattr("magformer.engine.trainer.dist.is_initialized", lambda: True)
    monkeypatch.setattr("magformer.engine.trainer.dist.barrier", lambda: None)

    metrics = trainer.evaluate()

    assert "val/segm_AP" in metrics
    assert "val/mAP" in metrics
    assert metrics["val/diag_num_eval_images"] == 1.0
    assert "val/loss" not in metrics
    assert logged["val/mAP"] == metrics["val/mAP"]
    assert checkpoint_calls == [True]


def test_single_gpu_and_ddp_evaluation_return_the_same_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    coco = _write_min_coco_dataset(tmp_path / "ds")
    model = _EvalOnlyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    shared_loader = [
        {
            "images": torch.zeros(1, 3, 32, 32),
            "depths": torch.zeros(1, 1, 32, 32),
            "image_ids": [1],
        }
    ]

    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        train_loader=shared_loader,
        val_loader=shared_loader,
        val_dataset=SimpleNamespace(coco=coco, category_ids=[1]),
        device=torch.device("cpu"),
        output_dir=str(tmp_path / "single"),
        max_iter=1,
        eval_period=1,
        checkpoint_period=100,
        log_period=1,
        amp_enabled=False,
    )
    trainer.current_iter = 3
    trainer.logger = SimpleNamespace(log_scalars=lambda *a, **k: None, close=lambda: None)
    monkeypatch.setattr(trainer, "_save_eval_visualization", lambda batch, outputs: None)
    monkeypatch.setattr(trainer, "save_checkpoint", lambda is_best=False: None)

    ddp_trainer = DDPTrainer.__new__(DDPTrainer)
    ddp_trainer.model = _DDPWrapper(_EvalOnlyModel())
    ddp_trainer.val_loader = shared_loader
    ddp_trainer.val_dataset = SimpleNamespace(coco=coco, category_ids=[1])
    ddp_trainer.device = torch.device("cpu")
    ddp_trainer.output_dir = tmp_path / "ddp"
    ddp_trainer.output_dir.mkdir(parents=True, exist_ok=True)
    ddp_trainer.visualization_dir = ddp_trainer.output_dir / "visualizations"
    ddp_trainer.visualization_dir.mkdir(parents=True, exist_ok=True)
    ddp_trainer.metrics_log_file = ddp_trainer.output_dir / "metrics_log.jsonl"
    ddp_trainer.metrics_csv_file = ddp_trainer.output_dir / "metrics_log.csv"
    ddp_trainer._csv_header_written = False
    ddp_trainer.rank = 0
    ddp_trainer.world_size = 2
    ddp_trainer.amp_enabled = False
    ddp_trainer.current_iter = 3
    ddp_trainer.best_metric = float("-inf")
    ddp_trainer.max_iter = 1
    ddp_trainer._iter_time_window_sec = deque(maxlen=20)
    ddp_trainer._train_start_monotonic = None
    ddp_trainer._pbar = None
    ddp_trainer.logger = SimpleNamespace(log_scalars=lambda *a, **k: None, close=lambda: None)
    monkeypatch.setattr(ddp_trainer, "_save_eval_visualization", lambda batch, outputs: None)
    monkeypatch.setattr(ddp_trainer, "save_checkpoint", lambda is_best=False: None)

    def _fake_all_gather_object(output, value):
        filler = [] if isinstance(value, list) else 0
        for idx in range(len(output)):
            output[idx] = filler
        output[0] = value

    monkeypatch.setattr("magformer.engine.eval_runtime.dist.is_available", lambda: True)
    monkeypatch.setattr("magformer.engine.eval_runtime.dist.is_initialized", lambda: True)
    monkeypatch.setattr("magformer.engine.eval_runtime.dist.get_rank", lambda: 0)
    monkeypatch.setattr("magformer.engine.eval_runtime.dist.get_world_size", lambda: 2)
    monkeypatch.setattr("magformer.engine.eval_runtime.dist.all_gather_object", _fake_all_gather_object)
    monkeypatch.setattr("magformer.engine.evaluator.dist.is_available", lambda: True)
    monkeypatch.setattr("magformer.engine.evaluator.dist.is_initialized", lambda: True)
    monkeypatch.setattr("magformer.engine.evaluator.dist.get_world_size", lambda: 2)
    monkeypatch.setattr("magformer.engine.evaluator.dist.all_gather_object", _fake_all_gather_object)
    monkeypatch.setattr("magformer.engine.trainer.dist.is_available", lambda: True)
    monkeypatch.setattr("magformer.engine.trainer.dist.is_initialized", lambda: True)
    monkeypatch.setattr("magformer.engine.trainer.dist.barrier", lambda: None)

    single_metrics = trainer.evaluate()
    ddp_metrics = ddp_trainer.evaluate()

    assert single_metrics == ddp_metrics


def test_trainer_saves_first_best_checkpoint_even_when_map_is_zero(
    monkeypatch,
    tmp_path: Path,
) -> None:
    coco = _write_min_coco_dataset(tmp_path / "ds")
    model = _WrongEvalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        train_loader=[{"images": torch.zeros(1, 3, 32, 32), "depths": torch.zeros(1, 1, 32, 32)}],
        val_loader=[
            {
                "images": torch.zeros(1, 3, 32, 32),
                "depths": torch.zeros(1, 1, 32, 32),
                "image_ids": [1],
            }
        ],
        val_dataset=SimpleNamespace(coco=coco),
        device=torch.device("cpu"),
        output_dir=str(tmp_path / "out"),
        max_iter=1,
        eval_period=1,
        checkpoint_period=100,
        log_period=1,
        amp_enabled=False,
    )
    trainer.current_iter = 1

    checkpoint_calls = []

    monkeypatch.setattr(trainer, "_save_eval_visualization", lambda batch, outputs: None)
    trainer.logger = SimpleNamespace(
        log_scalars=lambda main_tag, tag_scalar_dict, step: None,
        close=lambda: None,
    )
    monkeypatch.setattr(trainer, "save_checkpoint", lambda is_best=False: checkpoint_calls.append(bool(is_best)))

    metrics = trainer.evaluate()

    assert metrics["val/mAP"] == 0.0
    assert trainer.best_metric == 0.0
    assert checkpoint_calls == [True]


def test_trainer_treats_empty_predictions_as_zero_metric_and_saves_best_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    coco = _write_min_coco_dataset(tmp_path / "ds")
    model = _EmptyEvalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        train_loader=[{"images": torch.zeros(1, 3, 32, 32), "depths": torch.zeros(1, 1, 32, 32)}],
        val_loader=[
            {
                "images": torch.zeros(1, 3, 32, 32),
                "depths": torch.zeros(1, 1, 32, 32),
                "image_ids": [1],
            }
        ],
        val_dataset=SimpleNamespace(coco=coco),
        device=torch.device("cpu"),
        output_dir=str(tmp_path / "out"),
        max_iter=1,
        eval_period=1,
        checkpoint_period=100,
        log_period=1,
        amp_enabled=False,
    )
    trainer.current_iter = 1

    checkpoint_calls = []

    monkeypatch.setattr(trainer, "_save_eval_visualization", lambda batch, outputs: None)
    trainer.logger = SimpleNamespace(
        log_scalars=lambda main_tag, tag_scalar_dict, step: None,
        close=lambda: None,
    )
    monkeypatch.setattr(trainer, "save_checkpoint", lambda is_best=False: checkpoint_calls.append(bool(is_best)))

    metrics = trainer.evaluate()

    assert metrics["val/mAP"] == 0.0
    assert metrics["val/diag_num_predictions"] == 0.0
    assert trainer.best_metric == 0.0
    assert checkpoint_calls == [True]
