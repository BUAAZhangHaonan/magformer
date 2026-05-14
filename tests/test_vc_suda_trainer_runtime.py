from __future__ import annotations

import json
import os
import socket

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from magformer.engine.trainer import Trainer
from magformer.engine.vc_suda_trainer import VCSUDADDPTrainer, VCSUDATrainer


class _FakeLogger:
    def log_scalars(self, *args, **kwargs):
        pass

    def close(self):
        pass


class _TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(
        self,
        images,
        depths,
        targets=None,
        padding_masks=None,
        depth_noise_masks=None,
        return_features=False,
    ):
        self.calls.append(
            {
                "targets": targets,
                "padding_masks": padding_masks,
                "depth_noise_masks": depth_noise_masks,
                "return_features": return_features,
            }
        )
        if self.training and targets is None and not return_features:
            raise ValueError("training raw outputs require return_features=True")
        logits = self.weight * torch.ones(images.shape[0], 2, 2, device=images.device)
        masks = self.weight * torch.ones(images.shape[0], 2, images.shape[-2], images.shape[-1], device=images.device)
        outputs = {"pred_logits": logits, "pred_masks": masks}
        if return_features:
            outputs["features"] = self.weight * torch.ones(images.shape[0], 4, images.shape[-2], images.shape[-1], device=images.device)
            return outputs
        return {"total_loss": self.weight * images.sum() * 0.0 + self.weight}


class _TinyStageBWeightedCEStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        empty_weight = torch.ones(2)
        empty_weight[-1] = 0.1
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        images,
        depths,
        targets=None,
        padding_masks=None,
        depth_noise_masks=None,
        return_features=False,
    ):
        del depths, padding_masks, depth_noise_masks, return_features
        logits = self.scale * torch.ones(images.shape[0], 2, 2, device=images.device)
        target_classes = torch.full(
            logits.shape[:2],
            1,
            dtype=torch.int64,
            device=images.device,
        )
        if targets:
            for batch_index, target in enumerate(targets):
                labels = target.get("labels")
                if labels is not None and labels.numel() > 0:
                    target_classes[batch_index, 0] = labels[0]
        loss = F.cross_entropy(logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"total_loss": loss}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stage_b_ddp_weighted_ce_worker(rank: int, world_size: int, port: int, output_dir: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        model = _TinyStageBWeightedCEStudent().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        cfg = {"runtime": {"grad_accum_steps": 1}, "vc_suda": {"stage": "B"}}
        trainer = VCSUDADDPTrainer(
            model=model,
            criterion=_TinyCriterion(),
            optimizer=optimizer,
            train_loader=[],
            config=cfg,
            device=device,
            output_dir=os.path.join(output_dir, f"rank{rank}"),
            max_iter=1,
            log_period=100,
            amp_enabled=False,
            clip_gradients=False,
            logger_config={"type": "none"},
            vc_suda_config={"stage": "B"},
        )
        b, h, w = 1, 4, 4
        batch = {
            "source_images": torch.ones(b, 3, h, w),
            "source_depths": torch.ones(b, 1, h, w),
            "source_padding_masks": torch.zeros(b, h, w, dtype=torch.bool),
            "source_noise_masks": torch.zeros(b, 1, h, w),
            "source_annotations": [{"labels": torch.tensor([0]), "masks": torch.ones(1, h, w)}],
            "target_labeled_images": 2 * torch.ones(b, 3, h, w),
            "target_labeled_depths": 2 * torch.ones(b, 1, h, w),
            "target_labeled_padding_masks": torch.zeros(b, h, w, dtype=torch.bool),
            "target_labeled_noise_masks": torch.zeros(b, 1, h, w),
            "target_labeled_annotations": [
                {"labels": torch.tensor([0]), "masks": torch.ones(1, h, w)}
            ],
        }

        trainer._train_step(batch)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


class _TinyCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.supervised_criterion = object()

    def supervised_loss(self, outputs, targets):
        return {"total_loss": outputs["pred_logits"].sum() * 0.0 + 1.0}

    def pseudo_label_loss(self, outputs, pseudo_targets):
        return {"pseudo_total": outputs["pred_logits"].sum() * 0.0 + 0.5}


class _TinyTeacher:
    momentum = 0.99
    warmup_steps = 1

    def __init__(self):
        self.calls = []
        self.updates = []

    def __call__(self, images, depths, padding_masks=None, depth_noise_masks=None):
        self.calls.append({"padding_masks": padding_masks, "depth_noise_masks": depth_noise_masks})
        return {
            "pred_logits": torch.ones(images.shape[0], 2, 2, device=images.device),
            "pred_masks": torch.ones(images.shape[0], 2, images.shape[-2], images.shape[-1], device=images.device),
        }

    def update_ema(self, model, step):
        self.updates.append(step)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        self.loaded = state


class _TinyScorer:
    def score(self, teacher_outputs, depths):
        batch = teacher_outputs["pred_logits"].shape[0]
        h, w = teacher_outputs["pred_masks"].shape[-2:]
        return [
            {
                "labels": torch.tensor([0], device=depths.device),
                "masks": torch.ones(1, h, w, device=depths.device),
                "scores": torch.tensor([0.9], device=depths.device),
            }
            for _ in range(batch)
        ]

    def filter_by_threshold(self, scored, threshold):
        return scored


class _MixedScorer:
    def score(self, teacher_outputs, depths):
        h, w = teacher_outputs["pred_masks"].shape[-2:]
        device = depths.device
        return [
            {
                "labels": torch.tensor([0, 0], device=device),
                "masks": torch.ones(2, h, w, device=device),
                "scores": torch.tensor([0.9, 0.1], device=device),
            },
            {
                "labels": torch.tensor([0], device=device),
                "masks": torch.ones(1, h, w, device=device),
                "scores": torch.tensor([0.05], device=device),
            },
        ]

    def filter_by_threshold(self, scored, threshold):
        filtered = []
        for result in scored:
            keep = result["scores"] >= threshold
            filtered.append(
                {
                    "labels": result["labels"][keep],
                    "masks": result["masks"][keep],
                    "scores": result["scores"][keep],
                }
            )
        return filtered


class _TinyCurriculum:
    def __init__(self):
        self.epochs = []

    def get_threshold(self, epoch):
        self.epochs.append(epoch)
        return 0.5

    def get_unsupervised_weight(self, epoch, max_weight=1.0, warmup_epochs=10):
        return max_weight * min(1.0, float(epoch + 1) / float(max(1, warmup_epochs)))

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        self.loaded = state


def _patch_logger(monkeypatch):
    monkeypatch.setattr(VCSUDATrainer, "_setup_logger", lambda self, logger_config: _FakeLogger())


def _batch():
    b, h, w = 1, 4, 4
    return {
        "source_images": torch.ones(b, 3, h, w),
        "source_depths": torch.ones(b, 1, h, w),
        "source_padding_masks": torch.ones(b, h, w, dtype=torch.bool),
        "source_noise_masks": torch.ones(b, 1, h, w),
        "source_annotations": [{"labels": torch.tensor([0]), "masks": torch.ones(1, h, w), "boxes": torch.ones(1, 4)}],
        "target_weak_images": torch.ones(b, 3, h, w),
        "target_weak_depths": torch.ones(b, 1, h, w),
        "target_weak_padding_masks": torch.zeros(b, h, w, dtype=torch.bool),
        "target_weak_noise_masks": torch.zeros(b, 1, h, w),
        "target_strong_images": torch.ones(b, 3, h, w),
        "target_strong_depths": torch.ones(b, 1, h, w),
        "target_strong_padding_masks": torch.zeros(b, h, w, dtype=torch.bool),
        "target_strong_noise_masks": torch.zeros(b, 1, h, w),
    }


def _trainer(tmp_path, monkeypatch, **overrides):
    _patch_logger(monkeypatch)
    model = overrides.pop("model", _TinyStudent())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = {"runtime": {"grad_accum_steps": overrides.pop("grad_accum_steps", 1)}, "vc_suda": {}}
    vc_cfg = {
        "stage": overrides.pop("stage", "C"),
        "unsupervised_weight": 1.0,
        "unsupervised_warmup_epochs": 2,
        "target_labeled_weight": overrides.pop("target_labeled_weight", 1.0),
        "domain_adaptation": overrides.pop(
            "domain_adaptation",
            {"prototype_weight": 0.0, "boundary_weight": 0.0, "modality_dropout_weight": 0.0},
        ),
    }
    cfg["vc_suda"] = vc_cfg
    return VCSUDATrainer(
        model=model,
        criterion=overrides.pop("criterion", _TinyCriterion()),
        optimizer=optimizer,
        train_loader=[],
        config=cfg,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        max_iter=2,
        log_period=100,
        amp_enabled=False,
        clip_gradients=False,
        resume=overrides.pop("resume", None),
        logger_config={"type": "none"},
        ema_teacher=overrides.pop("ema_teacher", _TinyTeacher()),
        pseudo_label_scorer=overrides.pop("pseudo_label_scorer", _TinyScorer()),
        curriculum_scheduler=overrides.pop("curriculum_scheduler", _TinyCurriculum()),
        vc_suda_config=vc_cfg,
        domain_losses=overrides.pop("domain_losses", {}),
        **overrides,
    )


def test_vc_suda_trainer_resumes_after_vc_components_are_initialized(tmp_path, monkeypatch):
    model = _TinyStudent()
    ckpt = tmp_path / "resume.pth"
    torch.save(
        {
            "iter": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": torch.optim.SGD(model.parameters(), lr=0.1).state_dict(),
            "ema_teacher_state_dict": {},
            "curriculum_state_dict": {},
        },
        ckpt,
    )

    trainer = _trainer(tmp_path, monkeypatch, model=model, resume=str(ckpt))

    assert trainer.current_iter == 3
    assert hasattr(trainer.ema_teacher, "loaded")
    assert hasattr(trainer.curriculum_scheduler, "loaded")


def test_stage_c_resume_requires_ema_teacher_state_dict(tmp_path, monkeypatch):
    model = _TinyStudent()
    ckpt = tmp_path / "resume_without_ema.pth"
    torch.save(
        {
            "iter": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": torch.optim.SGD(model.parameters(), lr=0.1).state_dict(),
            "curriculum_state_dict": {},
        },
        ckpt,
    )

    with pytest.raises(ValueError, match="ema_teacher_state_dict"):
        _trainer(tmp_path, monkeypatch, model=model, resume=str(ckpt), stage="C")


def test_trainer_final_eval_accepts_dict_return_without_refinalizing(tmp_path, monkeypatch):
    monkeypatch.setattr(Trainer, "_setup_logger", lambda self, logger_config: _FakeLogger())
    monkeypatch.setattr(Trainer, "_console_log", lambda self, message: None)
    monkeypatch.setattr(Trainer, "save_checkpoint", lambda self, is_best=False: None)

    model = nn.Linear(1, 1)
    trainer = Trainer(
        model=model,
        criterion=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=[object()],
        config={"runtime": {}},
        device=torch.device("cpu"),
        output_dir=tmp_path,
        max_iter=1,
        eval_period=99999,
        checkpoint_period=99999,
        log_period=99999,
        amp_enabled=False,
    )

    def _train_step(self, batch):
        del batch
        self.current_iter += 1
        return {"total_loss": torch.tensor(0.0)}

    monkeypatch.setattr(Trainer, "_train_step", _train_step)
    monkeypatch.setattr(Trainer, "evaluate", lambda self: {"val/mAP": 0.5})

    trainer.train()


def test_vc_suda_train_step_passes_semi_collate_masks_and_advances_unsup_weight(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch)

    losses = trainer._train_step(_batch())

    assert losses["total_loss"].item() > 1.0
    assert trainer.ema_teacher.calls[0]["padding_masks"] is not None
    assert trainer.ema_teacher.calls[0]["depth_noise_masks"] is not None
    source_call = trainer.model.calls[0]
    target_call = trainer.model.calls[1]
    assert source_call["padding_masks"] is not None
    assert source_call["depth_noise_masks"] is not None
    assert target_call["padding_masks"] is not None
    assert target_call["depth_noise_masks"] is not None
    assert target_call["return_features"] is True


def test_pseudo_label_metrics_reports_count_empty_threshold_and_keep_rate():
    scored = [
        {"scores": torch.tensor([0.9, 0.3, 0.1])},
        {"scores": torch.tensor([0.7])},
        {"scores": torch.empty(0)},
    ]
    filtered = [
        {"scores": torch.tensor([0.9, 0.3])},
        {"scores": torch.empty(0)},
        {"scores": torch.empty(0)},
    ]

    metrics = VCSUDATrainer._compute_pseudo_label_metrics(
        scored, filtered, threshold=0.2
    )

    assert metrics["pseudo_kept_count"] == pytest.approx(2.0)
    assert metrics["pseudo_empty_images"] == pytest.approx(2.0)
    assert metrics["pseudo_threshold"] == pytest.approx(0.2)
    assert metrics["pseudo_keep_rate"] == pytest.approx(0.5)


def test_stage_c_train_step_logs_structured_pseudo_label_metrics(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch, pseudo_label_scorer=_MixedScorer())
    trainer.log_period = 1
    batch = _batch()
    b, h, w = 2, 4, 4
    batch.update(
        {
            "target_weak_images": torch.ones(b, 3, h, w),
            "target_weak_depths": torch.ones(b, 1, h, w),
            "target_weak_padding_masks": torch.zeros(b, h, w, dtype=torch.bool),
            "target_weak_noise_masks": torch.zeros(b, 1, h, w),
            "target_strong_images": torch.ones(b, 3, h, w),
            "target_strong_depths": torch.ones(b, 1, h, w),
            "target_strong_padding_masks": torch.zeros(b, h, w, dtype=torch.bool),
            "target_strong_noise_masks": torch.zeros(b, 1, h, w),
        }
    )

    losses = trainer._train_step(batch)

    assert losses["pseudo_kept_count"].item() == pytest.approx(1.0)
    assert losses["pseudo_empty_images"].item() == pytest.approx(1.0)
    assert losses["pseudo_threshold"].item() == pytest.approx(0.5)
    assert losses["pseudo_keep_rate"].item() == pytest.approx(1.0 / 3.0)

    payload = json.loads(
        (tmp_path / "metrics_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )
    assert payload["train/pseudo_kept_count"] == pytest.approx(1.0)
    assert payload["train/pseudo_empty_images"] == pytest.approx(1.0)
    assert payload["train/pseudo_threshold"] == pytest.approx(0.5)
    assert payload["train/pseudo_keep_rate"] == pytest.approx(1.0 / 3.0)


def test_vc_suda_modality_dropout_uses_raw_output_path_when_targets_are_none(tmp_path, monkeypatch):
    class _ConsistencyLoss(nn.Module):
        def forward(self, full_preds, dropped_preds):
            return (dropped_preds["pred_logits"] - full_preds["pred_logits"].detach()).pow(2).mean()

    trainer = _trainer(
        tmp_path,
        monkeypatch,
        stage="D",
        domain_adaptation={"prototype_weight": 0.0, "boundary_weight": 0.0, "modality_dropout_weight": 1.0, "modality_dropout_prob": 1.0},
        domain_losses={"modality_dropout": _ConsistencyLoss()},
    )

    trainer._train_step(_batch())

    dropped_call = trainer.model.calls[-1]
    assert dropped_call["targets"] is None
    assert dropped_call["return_features"] is True


def test_stage_b_train_step_uses_target_labeled_supervised_batch(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch, stage="B")
    batch = _batch()
    b, h, w = 1, 4, 4
    batch.update(
        {
            "target_labeled_images": 2 * torch.ones(b, 3, h, w),
            "target_labeled_depths": 2 * torch.ones(b, 1, h, w),
            "target_labeled_padding_masks": torch.zeros(b, h, w, dtype=torch.bool),
            "target_labeled_noise_masks": torch.zeros(b, 1, h, w),
            "target_labeled_annotations": [
                {
                    "labels": torch.tensor([0]),
                    "masks": torch.ones(1, h, w),
                    "boxes": torch.ones(1, 4),
                    "image_id": 25,
                }
            ],
        }
    )

    losses = trainer._train_step(batch)

    assert losses["total_loss"].item() == pytest.approx(2.0)
    assert len(trainer.model.calls) == 2
    target_labeled_call = trainer.model.calls[1]
    assert target_labeled_call["targets"][0]["image_id"] == 25
    assert target_labeled_call["padding_masks"] is not None
    assert target_labeled_call["depth_noise_masks"] is not None
    assert target_labeled_call["return_features"] is False


@pytest.mark.parametrize(
    ("target_labeled_weight", "expected_total"),
    [
        (0.0, 1.0),
        (0.5, 1.5),
        (1.0, 2.0),
    ],
)
def test_stage_b_target_labeled_weight_scales_total_loss(
    tmp_path, monkeypatch, target_labeled_weight, expected_total
):
    trainer = _trainer(
        tmp_path,
        monkeypatch,
        stage="B",
        target_labeled_weight=target_labeled_weight,
    )
    batch = _batch()
    b, h, w = 1, 4, 4
    batch.update(
        {
            "target_labeled_images": 2 * torch.ones(b, 3, h, w),
            "target_labeled_depths": 2 * torch.ones(b, 1, h, w),
            "target_labeled_padding_masks": torch.zeros(b, h, w, dtype=torch.bool),
            "target_labeled_noise_masks": torch.zeros(b, 1, h, w),
            "target_labeled_annotations": [
                {
                    "labels": torch.tensor([0]),
                    "masks": torch.ones(1, h, w),
                    "boxes": torch.ones(1, 4),
                }
            ],
        }
    )

    losses = trainer._train_step(batch)

    assert losses["total_loss"].item() == pytest.approx(expected_total)
    assert losses["source_total_loss"].item() == pytest.approx(1.0)
    assert losses["tl_total_loss_raw"].item() == pytest.approx(1.0)
    assert losses["tl_total_loss_weighted"].item() == pytest.approx(
        target_labeled_weight
    )
    assert losses["tl_weight"].item() == pytest.approx(target_labeled_weight)


def test_stage_b_target_labeled_weight_metrics_are_logged(tmp_path, monkeypatch):
    trainer = _trainer(
        tmp_path,
        monkeypatch,
        stage="B",
        target_labeled_weight=0.5,
    )
    trainer.log_period = 1
    batch = _batch()
    b, h, w = 1, 4, 4
    batch.update(
        {
            "target_labeled_images": 2 * torch.ones(b, 3, h, w),
            "target_labeled_depths": 2 * torch.ones(b, 1, h, w),
            "target_labeled_padding_masks": torch.zeros(b, h, w, dtype=torch.bool),
            "target_labeled_noise_masks": torch.zeros(b, 1, h, w),
            "target_labeled_annotations": [
                {
                    "labels": torch.tensor([0]),
                    "masks": torch.ones(1, h, w),
                    "boxes": torch.ones(1, 4),
                }
            ],
        }
    )

    trainer._train_step(batch)

    payload = json.loads(
        (tmp_path / "metrics_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )
    assert payload["train/source_total_loss"] == pytest.approx(1.0)
    assert payload["train/tl_total_loss_raw"] == pytest.approx(1.0)
    assert payload["train/tl_total_loss_weighted"] == pytest.approx(0.5)
    assert payload["train/tl_weight"] == pytest.approx(0.5)



def test_vc_suda_ddp_evaluate_forwards_runtime_eval_limits(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import magformer.engine.eval_runtime as eval_runtime

    captured = {}
    trainer = VCSUDADDPTrainer.__new__(VCSUDADDPTrainer)
    trainer.model = _FakeEvalWrapper(_TinyStudent())
    trainer.val_loader = [object()]
    trainer.val_dataset = SimpleNamespace(coco=object(), category_ids=[7])
    trainer.device = torch.device("cpu")
    trainer.output_dir = tmp_path
    trainer.amp_enabled = False
    trainer.rank = 0
    trainer.world_size = 1
    trainer.eval_iou_types = ["bbox"]
    trainer.eval_max_images = 28
    trainer._finalize_eval_result = lambda result: result.log_dict

    def _fake_run(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(log_dict={"val/bbox_AP": 0.5})

    monkeypatch.setattr(eval_runtime, "run_inference_evaluation", _fake_run)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    metrics = trainer.evaluate()

    assert captured["iou_types"] == ["bbox"]
    assert captured["max_images"] == 28
    assert captured["category_ids"] == [7]
    assert metrics == {"val/bbox_AP": 0.5}


class _FakeEvalWrapper:
    def __init__(self, module):
        self.module = module

    def eval(self):
        self.module.eval()
        return self

    def train(self, mode=True):
        self.module.train(mode)
        return self

def test_vc_suda_ddp_trainer_disables_buffer_broadcast(tmp_path, monkeypatch):
    captured = {}

    class _FakeDDP(nn.Module):
        def __init__(self, module, *args, **kwargs):
            super().__init__()
            del args
            self.module = module
            captured.update(kwargs)

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _FakeDDP)
    _patch_logger(monkeypatch)
    model = _TinyStudent()
    cfg = {"runtime": {"grad_accum_steps": 1}, "vc_suda": {"stage": "B"}}

    VCSUDADDPTrainer(
        model=model,
        criterion=_TinyCriterion(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=[],
        config=cfg,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        max_iter=1,
        log_period=100,
        amp_enabled=False,
        clip_gradients=False,
        logger_config={"type": "none"},
        vc_suda_config={"stage": "B"},
    )

    assert captured.get("broadcast_buffers") is False


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least two CUDA devices for NCCL DDP regression",
)
def test_stage_b_ddp_multi_forward_one_backward_does_not_mutate_ce_weight(tmp_path):
    if os.environ.get("MAGFORMER_RUN_DDP_REGRESSION") != "1":
        pytest.skip("set MAGFORMER_RUN_DDP_REGRESSION=1 to run the CUDA DDP regression")

    world_size = 2
    port = _find_free_port()
    torch.multiprocessing.spawn(
        _stage_b_ddp_weighted_ce_worker,
        args=(world_size, port, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )


def test_vc_suda_gradient_accumulation_delays_optimizer_step(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch, grad_accum_steps=2)
    before = trainer.model.weight.detach().clone()

    trainer._train_step(_batch())
    after_first = trainer.model.weight.detach().clone()
    trainer._train_step(_batch())
    after_second = trainer.model.weight.detach().clone()

    assert torch.equal(before, after_first)
    assert not torch.equal(after_first, after_second)


def test_vc_suda_uncertainty_weighting_must_be_optimizer_managed(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="UncertaintyWeighting"):
        _trainer(tmp_path, monkeypatch, use_uncertainty_weighting=True, uw_module=None)


def _run_plain_trainer_with_plateau_evals(tmp_path, monkeypatch, early_stop_cfg, max_iter=6):
    _patch_logger(monkeypatch)
    monkeypatch.setattr(Trainer, "_setup_logger", lambda self, logger_config: _FakeLogger())
    monkeypatch.setattr(Trainer, "_console_log", lambda self, message: None)
    monkeypatch.setattr(Trainer, "save_checkpoint", lambda self, is_best=False: None)
    model = nn.Linear(1, 1)

    trainer = Trainer(
        model=model,
        criterion=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=[object()] * max_iter,
        config={"runtime": {"early_stop": early_stop_cfg}},
        device=torch.device("cpu"),
        output_dir=tmp_path,
        max_iter=max_iter,
        eval_period=1,
        checkpoint_period=max_iter + 1,
        log_period=max_iter + 1,
        amp_enabled=False,
    )

    eval_calls = []

    def _train_step(self, batch):
        self.current_iter += 1
        return {"total_loss": torch.tensor(0.0)}

    def _evaluate(self):
        eval_calls.append(self.current_iter)
        metric = 0.5
        if metric > self.best_metric:
            self.best_metric = metric
        return {"val/mAP": metric}

    monkeypatch.setattr(Trainer, "_train_step", _train_step)
    monkeypatch.setattr(Trainer, "evaluate", _evaluate)
    trainer.train()
    return trainer, eval_calls


def test_plain_trainer_disables_early_stop_when_runtime_early_stop_is_none(tmp_path, monkeypatch):
    trainer, eval_calls = _run_plain_trainer_with_plateau_evals(
        tmp_path, monkeypatch, early_stop_cfg=None, max_iter=6
    )

    assert trainer.early_stop is False
    assert trainer.current_iter == 6
    assert len(eval_calls) == 6


def test_plain_trainer_enables_early_stop_only_when_explicitly_configured(tmp_path, monkeypatch):
    trainer, eval_calls = _run_plain_trainer_with_plateau_evals(
        tmp_path,
        monkeypatch,
        early_stop_cfg={"enabled": True, "patience": 5, "min_delta": 0.1, "target_ap": 70.0},
        max_iter=6,
    )

    assert trainer.early_stop is True
    assert trainer.current_iter == 5
    assert len(eval_calls) == 5
