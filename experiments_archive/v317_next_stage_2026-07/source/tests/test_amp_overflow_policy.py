from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from magformer.config import RuntimeConfig
from magformer.engine.trainer import Trainer


class _Sampler:
    seed = 23


class _Loader:
    batch_size = 1
    num_workers = 0
    prefetch_factor = None
    persistent_workers = False
    drop_last = False
    collate_fn = None

    def __init__(self) -> None:
        self.dataset = [0]
        self.sampler = _Sampler()

    def __iter__(self):
        return iter(())

    def state_dict(self):
        return {"cursor": 0}

    def load_state_dict(self, state_dict) -> None:
        assert state_dict == {"cursor": 0}

    def validate_resume_state(self, state_dict, **kwargs):
        del kwargs
        return dict(state_dict)

    def is_epoch_exhausted(self) -> bool:
        return False

    def start_next_epoch(self, epoch: int):
        del epoch
        return iter(self)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

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


class _FakeGradScaler:
    def __init__(self, init_scale: float = 8.0) -> None:
        self.scale_value = float(init_scale)
        self.found_inf = False

    def get_scale(self) -> float:
        return self.scale_value

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer) -> None:
        del optimizer

    def step(self, optimizer) -> None:
        self.found_inf = any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        if not self.found_inf:
            optimizer.step()

    def update(self) -> None:
        if self.found_inf:
            self.scale_value *= 0.5

    def state_dict(self):
        return {"scale_value": self.scale_value}

    def load_state_dict(self, state_dict) -> None:
        self.scale_value = float(state_dict["scale_value"])


def _batch():
    return {
        "images": torch.ones(1, 1),
        "depths": torch.ones(1, 1),
    }


def _trainer(tmp_path: Path, *, limit: int = 16) -> Trainer:
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        lr_scheduler=_CountingScheduler(),
        train_loader=_Loader(),
        config={
            "runtime": {
                "grad_accum_steps": 1,
                "amp_init_scale": 8.0,
                "max_consecutive_amp_skips": limit,
                "early_stop": {"enabled": False},
            }
        },
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=2,
        eval_period=2,
        checkpoint_period=2,
        log_period=100,
        amp_enabled=False,
        clip_gradients=True,
        clip_value=1.0,
        logger_config={"type": "none"},
    )
    trainer.amp_enabled = True
    trainer.scaler = _FakeGradScaler(trainer.amp_init_scale)
    return trainer


@pytest.fixture(autouse=True)
def _disable_cuda_autocast(monkeypatch):
    monkeypatch.setattr(
        "magformer.engine.trainer.autocast",
        lambda *args, **kwargs: nullcontext(),
    )


def _inject_infinite_gradient(model: _TinyModel):
    return model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))


def test_amp_overflow_skips_backs_off_and_persists_finite_telemetry(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = _trainer(tmp_path)
    weight_before = trainer.model.weight.detach().clone()
    hook = _inject_infinite_gradient(trainer.model)
    clip_calls = []
    monkeypatch.setattr(
        trainer,
        "_clip_gradients_with_known_norm",
        lambda grad_norm: clip_calls.append(grad_norm),
    )

    trainer._train_step(_batch())
    hook.remove()

    assert torch.equal(trainer.model.weight.detach(), weight_before)
    assert trainer.optimizer_step == 0
    assert trainer.amp_skipped_steps == 1
    assert trainer.consecutive_amp_skips == 1
    assert trainer.scaler.get_scale() == pytest.approx(4.0)
    assert trainer._last_amp_scale_before_step == pytest.approx(8.0)
    assert trainer._last_amp_scale_after_step == pytest.approx(4.0)
    assert trainer._last_grad_finite is False
    assert trainer._last_preclip_grad_norm is None
    assert trainer._last_nonfinite_grad_param == "weight"
    assert trainer._last_nonfinite_grad_count == 1
    assert trainer.lr_scheduler.calls == 0
    assert clip_calls == []

    trainer._append_metrics_log({"train/loss": 1.0}, phase="train")
    payload = json.loads(trainer.metrics_log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["preclip_grad_norm"] is None
    assert payload["grad_finite"] is False
    assert payload["amp_scale_before_step"] == pytest.approx(8.0)
    assert payload["amp_scale_after_step"] == pytest.approx(4.0)
    assert payload["first_nonfinite_grad_param"] == "weight"
    assert payload["nonfinite_grad_count"] == 1


def test_amp_success_after_overflow_resets_consecutive_counter(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    hook = _inject_infinite_gradient(trainer.model)
    trainer._train_step(_batch())
    hook.remove()

    trainer._train_step(_batch())

    assert trainer.optimizer_step == 1
    assert trainer.amp_skipped_steps == 1
    assert trainer.consecutive_amp_skips == 0
    assert trainer._last_grad_finite is True
    assert trainer._last_nonfinite_grad_param is None
    assert trainer._last_nonfinite_grad_count == 0
    assert trainer.lr_scheduler.calls == 1


def test_amp_consecutive_skip_limit_raises_with_diagnostics(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, limit=1)
    hook = _inject_infinite_gradient(trainer.model)
    trainer._train_step(_batch())

    with pytest.raises(
        FloatingPointError,
        match=r"consecutive=2.*first_nonfinite_param=weight.*scale_before=4.0",
    ):
        trainer._train_step(_batch())
    hook.remove()

    assert trainer.amp_skipped_steps == 2
    assert trainer.consecutive_amp_skips == 2
    assert trainer.optimizer_step == 0
    assert trainer.current_iter == 2
    assert (
        trainer.optimizer_step + trainer.amp_skipped_steps
        == trainer.current_iter // trainer.grad_accum_steps
    )


def test_amp_disabled_nonfinite_gradient_still_fails_fast(tmp_path: Path) -> None:
    model = _TinyModel()
    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        config={"runtime": {"grad_accum_steps": 1}},
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=1,
        eval_period=1,
        checkpoint_period=1,
        amp_enabled=False,
        logger_config={"type": "none"},
    )
    hook = _inject_infinite_gradient(model)
    with pytest.raises(FloatingPointError, match="preclip_grad_norm"):
        trainer._train_step(_batch())
    hook.remove()


def test_nonfinite_forward_loss_fails_before_scaler_backoff(tmp_path: Path, monkeypatch) -> None:
    trainer = _trainer(tmp_path)
    monkeypatch.setattr(
        trainer.model,
        "forward",
        lambda *args, **kwargs: {"total_loss": trainer.model.weight * torch.tensor(float("inf"))},
    )

    with pytest.raises(FloatingPointError, match="Loss total_loss"):
        trainer._train_step(_batch())

    assert trainer.amp_skipped_steps == 0
    assert trainer.consecutive_amp_skips == 0
    assert trainer.scaler.get_scale() == pytest.approx(8.0)


def test_amp_consecutive_skip_state_resumes_exactly(tmp_path: Path) -> None:
    torch.manual_seed(19)
    uninterrupted = _trainer(tmp_path / "uninterrupted")
    hook = _inject_infinite_gradient(uninterrupted.model)
    uninterrupted._train_step(_batch())
    hook.remove()
    uninterrupted._train_step(_batch())

    torch.manual_seed(19)
    source = _trainer(tmp_path / "source")
    hook = _inject_infinite_gradient(source.model)
    source._train_step(_batch())
    hook.remove()
    source.save_checkpoint()
    # The skipped AMP attempt advances micro_step but not the optimizer-step
    # checkpoint budget.
    checkpoint = tmp_path / "source" / "checkpoint_iter_0000000.pth"

    torch.manual_seed(999)
    resumed = _trainer(tmp_path / "resumed")
    resumed.resume(str(checkpoint))
    assert resumed.amp_skipped_steps == 1
    assert resumed.consecutive_amp_skips == 1
    assert resumed.scaler.get_scale() == pytest.approx(4.0)
    resumed._train_step(_batch())

    assert resumed.current_iter == uninterrupted.current_iter == 2
    assert resumed.optimizer_step == uninterrupted.optimizer_step == 1
    assert resumed.amp_skipped_steps == uninterrupted.amp_skipped_steps == 1
    assert resumed.consecutive_amp_skips == uninterrupted.consecutive_amp_skips == 0
    assert resumed.scaler.state_dict() == uninterrupted.scaler.state_dict()
    assert resumed.lr_scheduler.state_dict() == uninterrupted.lr_scheduler.state_dict()
    assert torch.equal(resumed.model.weight, uninterrupted.model.weight)
    assert resumed.optimizer.state_dict() == uninterrupted.optimizer.state_dict()


def test_amp_runtime_policy_schema_defaults_and_validation() -> None:
    runtime = RuntimeConfig()
    assert runtime.amp_init_scale == pytest.approx(65536.0)
    assert runtime.max_consecutive_amp_skips == 16

    for invalid_scale in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            RuntimeConfig(amp_init_scale=invalid_scale)
    with pytest.raises(ValidationError):
        RuntimeConfig(max_consecutive_amp_skips=-1)


def test_large_finite_gradients_keep_a_finite_norm_and_clip(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, 3.0e38)
    trainer.model = model

    grad_norm = trainer._preclip_gradient_norm()

    assert math.isfinite(grad_norm)
    assert grad_norm > torch.finfo(torch.float32).max
    trainer._clip_gradients_with_known_norm(grad_norm)
    assert torch.isfinite(model.weight.grad).all()
    assert torch.linalg.vector_norm(model.weight.grad.double()).item() == pytest.approx(
        trainer.clip_value, rel=1e-5
    )
