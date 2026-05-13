from __future__ import annotations

import pytest
import torch
from torch import nn

from magformer.engine.trainer import Trainer
from magformer.engine.vc_suda_trainer import VCSUDATrainer


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


def test_plain_trainer_accepts_runtime_early_stop_none(tmp_path, monkeypatch):
    _patch_logger(monkeypatch)
    monkeypatch.setattr(Trainer, "_setup_logger", lambda self, logger_config: _FakeLogger())
    model = nn.Linear(1, 1)

    trainer = Trainer(
        model=model,
        criterion=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=[],
        config={"runtime": {"early_stop": None}},
        device=torch.device("cpu"),
        output_dir=tmp_path,
        amp_enabled=False,
    )

    assert trainer._patience_limit == 5
