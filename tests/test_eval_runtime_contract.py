from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from collections import deque

import torch

from magformer.engine.trainer import DDPTrainer, Trainer
from pycocotools.coco import COCO


def _write_min_coco_dataset(root: Path) -> COCO:
    (root / "images" / "val").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)

    import cv2
    import numpy as np

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "images" / "val" / "0001.png"), image)

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
    assert "val/loss" not in metrics
    assert "val/segm_AP" in logged
    assert "val/loss" not in logged
    assert trainer.best_metric == metrics["val/mAP"]
    assert checkpoint_calls == [True]


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
    assert "val/loss" not in metrics
    assert logged["val/mAP"] == metrics["val/mAP"]
    assert checkpoint_calls == [True]
