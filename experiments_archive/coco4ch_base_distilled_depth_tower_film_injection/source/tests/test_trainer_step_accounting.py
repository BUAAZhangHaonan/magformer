from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from magformer.config import load_config
from magformer.config.schema import SolverConfig
from magformer.engine.trainer import Trainer


class _StaticSampler:
    seed = 17


class _StaticStatefulLoader:
    batch_size = 1
    num_workers = 0
    prefetch_factor = None
    persistent_workers = False
    drop_last = False
    collate_fn = None

    def __init__(self, batch) -> None:
        self.dataset = [batch]
        self.sampler = _StaticSampler()
        self.cursor = 0

    def __iter__(self):
        return iter(self.dataset)

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])

    def validate_resume_state(self, state_dict, **kwargs):
        del kwargs
        return dict(state_dict)

    def is_epoch_exhausted(self) -> bool:
        return True

    def start_next_epoch(self, epoch: int):
        del epoch
        return iter(self)


class _TinyLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.optimizer_step_hooks = 0

    def on_successful_optimizer_step(self) -> None:
        self.optimizer_step_hooks += 1

    def forward(
        self,
        images,
        depths,
        targets=None,
        padding_masks=None,
        depth_noise_masks=None,
    ):
        del depths, targets, padding_masks, depth_noise_masks
        return {"total_loss": (self.weight * images.mean()).square()}


class _CountingScheduler:
    def __init__(self) -> None:
        self.calls = 0

    def step(self) -> None:
        self.calls += 1

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state_dict) -> None:
        self.calls = int(state_dict["calls"])


class _CountingEMA:
    def __init__(self) -> None:
        self.steps = []
        self.decay = 0.8
        self.warmup_iters = 2

    def update(self, step, model) -> None:
        del model
        self.steps.append(int(step))

    def state_dict(self):
        return {"steps": list(self.steps)}

    def load_state_dict(self, state_dict) -> None:
        self.steps = [int(step) for step in state_dict["steps"]]


def _batch():
    return {
        "images": torch.ones(1, 1),
        "depths": torch.ones(1, 1),
    }


def test_iteration_unit_schema_defaults_to_optimizer_steps() -> None:
    assert SolverConfig().iteration_unit == "optimizer_step"
    with pytest.raises(ValueError):
        SolverConfig(iteration_unit="micro_step")


def _trainer(
    tmp_path: Path,
    *,
    max_iter: int = 4,
    grad_accum_steps: int = 4,
    iteration_unit: str = "optimizer_step",
    eval_period: int = 2,
    checkpoint_period: int = 4,
    log_period: int = 100,
) -> Trainer:
    model = _TinyLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _CountingScheduler()
    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=_StaticStatefulLoader(_batch()),
        config={
            "solver": {"iteration_unit": iteration_unit},
            "runtime": {"grad_accum_steps": grad_accum_steps},
        },
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=max_iter,
        eval_period=eval_period,
        checkpoint_period=checkpoint_period,
        log_period=log_period,
        amp_enabled=False,
        clip_gradients=False,
        logger_config={"type": "none"},
    )
    trainer.ema = _CountingEMA()
    return trainer


def test_gradient_accumulation_advances_only_real_optimizer_steps(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    weight_before = trainer.model.weight.detach().clone()

    trainer._train_step(_batch())

    assert trainer.current_iter == 1
    assert trainer.optimizer_step == 0
    assert trainer._accum_count == 1
    assert trainer.lr_scheduler.calls == 0
    assert trainer.ema.steps == []
    assert trainer.model.optimizer_step_hooks == 0
    assert torch.equal(trainer.model.weight.detach(), weight_before)

    for _ in range(3):
        trainer._train_step(_batch())

    assert trainer.current_iter == 4
    assert trainer.optimizer_step == 1
    assert trainer._accum_count == 0
    assert trainer.lr_scheduler.calls == 1
    assert trainer.ema.steps == [0]
    assert trainer.model.optimizer_step_hooks == 1
    assert math.isfinite(trainer._last_preclip_grad_norm)
    assert not torch.equal(trainer.model.weight.detach(), weight_before)


def test_amp_skipped_attempt_does_not_advance_scheduler_or_ema(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = _trainer(tmp_path)
    monkeypatch.setattr(trainer, "_step_optimizer", lambda: False)

    for _ in range(4):
        trainer._train_step(_batch())

    assert trainer.current_iter == 4
    assert trainer.optimizer_step == 0
    assert trainer.amp_skipped_steps == 1
    assert trainer.lr_scheduler.calls == 0
    assert trainer.ema.steps == []
    assert trainer.model.optimizer_step_hooks == 0


def test_optimizer_step_budget_drives_cadence_and_ignores_amp_skip(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = _trainer(
        tmp_path,
        max_iter=2,
        eval_period=1,
        checkpoint_period=2,
        log_period=1,
    )
    eval_steps = []
    checkpoint_steps = []
    log_steps = []
    optimizer_attempts = iter([False, True, True])
    monkeypatch.setattr(trainer, "_step_optimizer", lambda: next(optimizer_attempts))

    def _evaluate():
        assert trainer._accum_count == 0
        eval_steps.append((trainer.current_iter, trainer.optimizer_step))
        return {}

    def _save_checkpoint(is_best=False):
        del is_best
        assert trainer._accum_count == 0
        checkpoint_steps.append((trainer.current_iter, trainer.optimizer_step))

    def _log_training(losses):
        del losses
        log_steps.append((trainer.current_iter, trainer.optimizer_step))

    monkeypatch.setattr(trainer, "evaluate", _evaluate)
    monkeypatch.setattr(trainer, "save_checkpoint", _save_checkpoint)
    monkeypatch.setattr(trainer, "_log_training", _log_training)

    trainer.train()

    assert trainer.optimizer_step == 2
    assert trainer.current_iter == 12
    assert trainer.amp_skipped_steps == 1
    assert trainer.lr_scheduler.calls == 2
    assert trainer.ema.steps == [0, 1]
    assert trainer.model.optimizer_step_hooks == 2
    assert log_steps == [(8, 1), (12, 2)]
    assert eval_steps == [(8, 1), (12, 2)]
    # One periodic checkpoint and one final checkpoint share optimizer step 2.
    assert checkpoint_steps == [(12, 2), (12, 2)]


def test_checkpoint_v3_round_trips_step_state(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path / "source")
    for _ in range(4):
        trainer._train_step(_batch())
    trainer.save_checkpoint()
    checkpoint = tmp_path / "source" / "checkpoint_iter_0000001.pth"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["iteration_unit"] == "optimizer_step"
    assert payload["iter"] == 1
    assert payload["micro_step"] == 4
    assert payload["optimizer_step"] == 1

    model = _TinyLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    resumed = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        lr_scheduler=_CountingScheduler(),
        train_loader=_StaticStatefulLoader(_batch()),
        config={
            "solver": {"iteration_unit": "optimizer_step"},
            "runtime": {"grad_accum_steps": 4},
        },
        device=torch.device("cpu"),
        output_dir=str(tmp_path / "resumed"),
        max_iter=4,
        eval_period=2,
        checkpoint_period=4,
        log_period=100,
        amp_enabled=False,
        clip_gradients=False,
        logger_config={"type": "none"},
    )
    resumed.ema = _CountingEMA()
    resumed.resume(str(checkpoint))

    assert resumed.current_iter == 4
    assert resumed.start_iter == 4
    assert resumed.optimizer_step == 1
    assert resumed.amp_skipped_steps == 0
    assert resumed.consecutive_amp_skips == 0
    assert resumed._accum_count == 0


def test_pre_unit_training_state_requires_explicit_legacy_mode(
    tmp_path: Path,
) -> None:
    source = _trainer(
        tmp_path / "source",
        max_iter=8,
        grad_accum_steps=4,
        iteration_unit="legacy_micro_step",
        eval_period=4,
        checkpoint_period=8,
    )
    for _ in range(4):
        source._train_step(_batch())
    source.save_checkpoint()
    source_path = tmp_path / "source" / "checkpoint_iter_0000004.pth"
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    checkpoint.pop("iteration_unit")
    legacy_path = tmp_path / "pre_unit_v3.pth"
    torch.save(checkpoint, legacy_path)

    optimizer_mode = _trainer(
        tmp_path / "optimizer-mode",
        max_iter=8,
        grad_accum_steps=4,
        iteration_unit="optimizer_step",
        eval_period=4,
        checkpoint_period=8,
    )
    with pytest.raises(RuntimeError, match="missing iteration_unit"):
        optimizer_mode.resume(str(legacy_path))

    legacy_mode = _trainer(
        tmp_path / "legacy-mode",
        max_iter=8,
        grad_accum_steps=4,
        iteration_unit="legacy_micro_step",
        eval_period=4,
        checkpoint_period=8,
    )
    legacy_mode.resume(str(legacy_path))
    assert legacy_mode.current_iter == 4
    assert legacy_mode.optimizer_step == 1


@pytest.mark.parametrize("grad_accum_steps", [1, 2])
def test_format_v1_checkpoint_rejects_full_resume(
    tmp_path: Path, grad_accum_steps: int
) -> None:
    model = _TinyLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = tmp_path / "legacy.pth"
    torch.save(
        {
            "iter": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint,
    )

    resumed_model = _TinyLossModel()
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
    with pytest.raises(RuntimeError, match="checkpoint_format_version < 3"):
        Trainer(
            model=resumed_model,
            criterion=None,
            optimizer=resumed_optimizer,
            train_loader=_StaticStatefulLoader(_batch()),
            config={"runtime": {"grad_accum_steps": grad_accum_steps}},
            device=torch.device("cpu"),
            output_dir=str(tmp_path / f"rejected-{grad_accum_steps}"),
            max_iter=4,
            eval_period=2,
            checkpoint_period=4,
            log_period=100,
            amp_enabled=False,
            clip_gradients=False,
            resume=str(checkpoint),
            logger_config={"type": "none"},
        )


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/v102_swins_mbv3l_8dec_32k_from_v88.yaml",
        "configs/finetune_1k_full_1536_v19.yaml",
    ],
    ids=["v102", "v19"],
)
def test_disabled_eval_period_sentinels_are_config_compatible(
    tmp_path: Path, config_path: str
) -> None:
    config = load_config(config_path)
    model = _TinyLossModel()

    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        config={"runtime": config.runtime.model_dump()},
        device=torch.device("cpu"),
        output_dir=str(tmp_path / Path(config_path).stem),
        max_iter=config.solver.max_iter,
        eval_period=config.runtime.eval_period,
        checkpoint_period=config.runtime.checkpoint_period,
        logger_config={"type": "none"},
    )

    assert trainer.eval_period > trainer.max_iter
    assert trainer.eval_period % trainer.grad_accum_steps != 0


def test_ema_warmup_uses_zero_based_real_decay_boundaries(tmp_path: Path) -> None:
    model = _TinyLossModel()
    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=[_batch()],
        config={
            "solver": {"iteration_unit": "optimizer_step"},
            "runtime": {
                "grad_accum_steps": 2,
                "ema_enabled": True,
                "ema_decay": 0.8,
                "ema_warmup_iters": 2,
            }
        },
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=6,
        eval_period=7,
        checkpoint_period=7,
        log_period=100,
        amp_enabled=False,
        clip_gradients=False,
        logger_config={"type": "none"},
    )

    assert trainer.ema is not None
    assert trainer.ema.warmup_iters == 2

    trainer._train_step(_batch())
    trainer._train_step(_batch())
    assert trainer.optimizer_step == 1
    assert trainer.ema.get_decay(0) == pytest.approx(0.0)
    assert trainer.model.weight.item() == pytest.approx(0.8)
    assert trainer.ema.shadow["weight"].item() == pytest.approx(0.8)

    trainer._train_step(_batch())
    trainer._train_step(_batch())
    assert trainer.optimizer_step == 2
    assert trainer.ema.get_decay(1) == pytest.approx(0.4)
    assert trainer.model.weight.item() == pytest.approx(0.64)
    assert trainer.ema.shadow["weight"].item() == pytest.approx(0.704)

    trainer._train_step(_batch())
    trainer._train_step(_batch())
    assert trainer.optimizer_step == 3
    assert trainer.ema.get_decay(2) == pytest.approx(0.8)
    assert trainer.model.weight.item() == pytest.approx(0.512)
    assert trainer.ema.shadow["weight"].item() == pytest.approx(0.6656)


def test_trainer_rejects_non_boundary_training_schedule(tmp_path: Path) -> None:
    model = _TinyLossModel()
    with pytest.raises(ValueError, match="max_iter=3 must be divisible"):
        Trainer(
            model=model,
            criterion=None,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            config={
                "solver": {"iteration_unit": "legacy_micro_step"},
                "runtime": {"grad_accum_steps": 2},
            },
            device=torch.device("cpu"),
            output_dir=str(tmp_path),
            max_iter=3,
            eval_period=2,
            checkpoint_period=4,
            logger_config={"type": "none"},
        )
