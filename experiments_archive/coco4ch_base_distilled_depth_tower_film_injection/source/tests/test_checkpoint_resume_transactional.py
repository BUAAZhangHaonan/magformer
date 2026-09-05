from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

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

    def __init__(self) -> None:
        self.dataset = [0]
        self.sampler = _StaticSampler()
        self.cursor = 0

    def __iter__(self):
        return iter(())

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


class _StatefulComponent:
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state_dict) -> None:
        self.value = int(state_dict["value"])
        if state_dict.get("raise_after_load", False):
            raise RuntimeError("injected component load failure")

    def step(self) -> None:
        pass


def _trainer(tmp_path: Path, *, source: bool) -> Trainer:
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(9.0 if source else 1.0)
        model.bias.fill_(8.0 if source else 0.0)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.9,
    )
    if not source:
        optimizer.param_groups[0]["lr"] = 0.1
    scheduler = _StatefulComponent(11 if source else -11)
    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=_StaticStatefulLoader(),
        config={
            "solver": {"iteration_unit": "legacy_micro_step"},
            "runtime": {
                "grad_accum_steps": 2,
                "early_stop": {"enabled": False},
            }
        },
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=4,
        eval_period=2,
        checkpoint_period=4,
        log_period=100,
        amp_enabled=False,
        clip_gradients=False,
        logger_config={"type": "none"},
    )
    trainer.amp_enabled = True
    trainer.scaler = _StatefulComponent(22 if source else -22)
    trainer.ema = _StatefulComponent(33 if source else -33)
    trainer.ema.decay = 0.8
    trainer.ema.warmup_iters = 2

    if source:
        trainer.current_iter = 4
        trainer.start_iter = 4
        trainer.optimizer_step = 2
        trainer.amp_skipped_steps = 0
        trainer.best_metric = 0.5
    return trainer


def _checkpoint(tmp_path: Path):
    source = _trainer(tmp_path / "source", source=True)
    source.save_checkpoint()
    path = tmp_path / "source" / "checkpoint_iter_0000004.pth"
    return torch.load(path, map_location="cpu", weights_only=True)


def _snapshot(trainer: Trainer):
    return {
        "model": {
            key: value.detach().clone()
            for key, value in trainer.model.state_dict().items()
        },
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "scheduler": copy.deepcopy(trainer.lr_scheduler.state_dict()),
        "scaler": copy.deepcopy(trainer.scaler.state_dict()),
        "ema": copy.deepcopy(trainer.ema.state_dict()),
        "rank_state": trainer._capture_rank_state(),
        "trainer": (
            trainer.current_iter,
            trainer.start_iter,
            trainer.optimizer_step,
            trainer.amp_skipped_steps,
            trainer.consecutive_amp_skips,
            trainer._accum_count,
            trainer.best_metric,
            trainer.early_stop_best_metric,
            trainer._patience_counter,
            trainer.early_stop,
        ),
    }


def _assert_nested_equal(actual, expected) -> None:
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, Mapping):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _reject_without_mutation(
    tmp_path: Path,
    checkpoint,
    *,
    match: str,
    exception_type=RuntimeError,
) -> None:
    path = tmp_path / "malformed.pth"
    torch.save(checkpoint, path)
    destination = _trainer(tmp_path / "destination", source=False)
    before = _snapshot(destination)

    with pytest.raises(exception_type, match=match):
        destination.resume(str(path))

    _assert_nested_equal(_snapshot(destination), before)


@pytest.mark.parametrize(
    "field_name",
    [
        "checkpoint_format_version",
        "grad_accum_steps",
        "accum_count",
        "iter",
        "micro_step",
        "optimizer_step",
        "amp_skipped_steps",
        "consecutive_amp_skips",
    ],
)
@pytest.mark.parametrize("invalid_type", ["bool", "float", "str"])
def test_resume_rejects_non_exact_integer_metadata_transactionally(
    tmp_path: Path, field_name: str, invalid_type: str
) -> None:
    checkpoint = _checkpoint(tmp_path)
    original = checkpoint[field_name]
    checkpoint[field_name] = {
        "bool": bool(original),
        "float": float(original),
        "str": str(original),
    }[invalid_type]

    _reject_without_mutation(
        tmp_path,
        checkpoint,
        match=f"{field_name} must have type int",
    )


@pytest.mark.parametrize("version", [1, 2, 4])
def test_resume_requires_exact_checkpoint_version_three(
    tmp_path: Path, version: int
) -> None:
    checkpoint = _checkpoint(tmp_path)
    checkpoint["checkpoint_format_version"] = version
    _reject_without_mutation(tmp_path, checkpoint, match="must equal 3")


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"grad_accum_steps": 0}, "grad_accum_steps must be positive"),
        ({"grad_accum_steps": 1}, "does not match"),
        ({"accum_count": -1}, "accum_count must be non-negative"),
        ({"iter": -1}, "iter must be non-negative"),
        ({"micro_step": -1}, "micro_step must be non-negative"),
        ({"optimizer_step": -1}, "optimizer_step must be non-negative"),
        ({"amp_skipped_steps": -1}, "amp_skipped_steps must be non-negative"),
        ({"accum_count": 1}, "inside a gradient-accumulation window"),
        ({"iter": 2}, "iter and micro_step fields disagree"),
        ({"iter": 3, "micro_step": 3}, "not on a"),
        ({"iter": 6, "micro_step": 6, "optimizer_step": 3}, "exceeds max_iter"),
        ({"optimizer_step": 1}, "accounting is inconsistent"),
    ],
)
def test_resume_rejects_invalid_step_invariants_transactionally(
    tmp_path: Path, updates, match: str
) -> None:
    checkpoint = _checkpoint(tmp_path)
    checkpoint.update(updates)
    _reject_without_mutation(tmp_path, checkpoint, match=match)


@pytest.mark.parametrize(
    "state_key",
    [
        "model_state_dict",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "scaler_state_dict",
        "ema_state_dict",
    ],
)
def test_resume_requires_every_enabled_component_state_transactionally(
    tmp_path: Path, state_key: str
) -> None:
    checkpoint = _checkpoint(tmp_path)
    del checkpoint[state_key]
    _reject_without_mutation(tmp_path, checkpoint, match=f"missing {state_key}")


@pytest.mark.parametrize(
    "state_key",
    [
        "model_state_dict",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "scaler_state_dict",
        "ema_state_dict",
    ],
)
def test_resume_requires_component_states_to_be_mappings_transactionally(
    tmp_path: Path, state_key: str
) -> None:
    checkpoint = _checkpoint(tmp_path)
    checkpoint[state_key] = []
    _reject_without_mutation(tmp_path, checkpoint, match=f"{state_key} must be a mapping")


def test_resume_validates_early_stop_state_before_any_mutation(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    checkpoint["early_stop_state"]["patience_counter"] = 1
    _reject_without_mutation(tmp_path, checkpoint, match="disabled early-stop state")


def test_valid_v3_checkpoint_restores_all_components_at_max_iter(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "valid.pth"
    torch.save(checkpoint, path)
    destination = _trainer(tmp_path / "destination", source=False)

    destination.resume(str(path))

    assert destination.current_iter == destination.max_iter == 4
    assert destination.start_iter == 4
    assert destination.optimizer_step == 2
    assert destination.amp_skipped_steps == 0
    assert destination.best_metric == pytest.approx(0.5)
    assert destination.model.weight.item() == pytest.approx(9.0)
    assert destination.model.bias.item() == pytest.approx(8.0)
    assert destination.optimizer.param_groups[0]["lr"] == pytest.approx(0.9)
    assert destination.lr_scheduler.value == 11
    assert destination.scaler.value == 22
    assert destination.ema.value == 33


@pytest.mark.parametrize("corruption", ["missing-key", "shape-mismatch"])
def test_model_load_failure_rolls_back_every_state(
    tmp_path: Path, corruption: str
) -> None:
    checkpoint = _checkpoint(tmp_path)
    if corruption == "missing-key":
        del checkpoint["model_state_dict"]["bias"]
        match = "Missing key"
    else:
        checkpoint["model_state_dict"]["weight"] = torch.full((2, 1), 9.0)
        match = "size mismatch"

    _reject_without_mutation(tmp_path, checkpoint, match=match)


def test_optimizer_load_failure_rolls_back_every_state(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    checkpoint["optimizer_state_dict"]["param_groups"] = []

    _reject_without_mutation(
        tmp_path,
        checkpoint,
        match="different number of parameter groups",
        exception_type=ValueError,
    )


@pytest.mark.parametrize(
    "state_key",
    [
        "lr_scheduler_state_dict",
        "scaler_state_dict",
        "ema_state_dict",
    ],
)
def test_component_load_failure_rolls_back_every_state(
    tmp_path: Path, state_key: str
) -> None:
    checkpoint = _checkpoint(tmp_path)
    checkpoint[state_key] = {
        "value": 999,
        "raise_after_load": True,
    }

    _reject_without_mutation(
        tmp_path,
        checkpoint,
        match="injected component load failure",
    )
