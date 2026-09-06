from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from magformer.config import EarlyStopConfig, RuntimeConfig, load_config
from magformer.engine.eval_runtime import EvaluationResult
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


def _policy(**overrides):
    policy = {
        "enabled": True,
        "monitor": "val/mAP",
        "patience": 2,
        "min_delta": 0.01,
        "target": None,
        "min_optimizer_step": 0,
    }
    policy.update(overrides)
    return policy


def _trainer(
    tmp_path: Path,
    *,
    early_stop=None,
    resume: str | None = None,
) -> Trainer:
    model = torch.nn.Linear(1, 1)
    return Trainer(
        model=model,
        criterion=None,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=_StaticStatefulLoader(),
        config={
            "runtime": {
                "grad_accum_steps": 1,
                "early_stop": (
                    {"enabled": False} if early_stop is None else early_stop
                ),
            }
        },
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=8,
        eval_period=2,
        checkpoint_period=2,
        log_period=100,
        amp_enabled=False,
        clip_gradients=False,
        resume=resume,
        logger_config={"type": "none"},
    )


def test_schema_defaults_disabled_and_warns_for_legacy_target() -> None:
    assert RuntimeConfig().early_stop.enabled is False

    with pytest.warns(FutureWarning, match="target_ap is deprecated"):
        policy = EarlyStopConfig(target_ap=70.0, patience=5, min_delta=0.1)

    assert policy.enabled is False
    assert policy.target == 70.0

    with pytest.raises(ValidationError, match="target must be in"):
        EarlyStopConfig(enabled=True, monitor="val/mAP", target=70.0)
    with pytest.raises(ValidationError):
        EarlyStopConfig(enabled=True, patience=0)
    with pytest.raises(ValidationError):
        EarlyStopConfig(enabled=True, unknown_option=True)
    with pytest.raises(ValidationError):
        EarlyStopConfig(enabled=False, min_delta=float("inf"))
    with pytest.raises(ValidationError):
        EarlyStopConfig(enabled=False, target=float("nan"))


def test_v317_legacy_policy_loads_with_warning_but_stays_disabled() -> None:
    with pytest.warns(FutureWarning, match="remains disabled"):
        config = load_config("configs/v317_init_from_m2f_swin_t.yaml")

    assert config.runtime.early_stop.enabled is False
    assert config.runtime.early_stop.target == pytest.approx(0.9)


def test_disabled_policy_never_mutates_early_stop_state(tmp_path: Path) -> None:
    trainer = _trainer(
        tmp_path,
        early_stop=_policy(enabled=False, patience=1, target=0.0),
    )
    trainer.optimizer_step = 100

    assert trainer._update_early_stopping({"val/mAP": 0.0}) is False
    assert trainer.early_stop_best_metric == float("-inf")
    assert trainer._patience_counter == 0


def test_significant_improvement_uses_independent_best_and_resets_patience(
    tmp_path: Path,
) -> None:
    trainer = _trainer(tmp_path, early_stop=_policy())

    assert trainer._update_early_stopping({"val/mAP": 0.50}) is False
    assert trainer.early_stop_best_metric == pytest.approx(0.50)
    assert trainer._patience_counter == 0

    assert trainer._update_early_stopping({"val/mAP": 0.505}) is False
    assert trainer.early_stop_best_metric == pytest.approx(0.50)
    assert trainer._patience_counter == 1

    assert trainer._update_early_stopping({"val/mAP": 0.511}) is False
    assert trainer.early_stop_best_metric == pytest.approx(0.511)
    assert trainer._patience_counter == 0
    assert trainer.best_metric == float("-inf")


def test_plateau_stops_exactly_at_patience(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, early_stop=_policy(patience=2))

    assert trainer._update_early_stopping({"val/mAP": 0.50}) is False
    assert trainer._update_early_stopping({"val/mAP": 0.505}) is False
    assert trainer._patience_counter == 1
    assert trainer._update_early_stopping({"val/mAP": 0.504}) is True
    assert trainer._patience_counter == 2
    assert trainer.early_stop is True


def test_target_stops_immediately(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, early_stop=_policy(target=0.8, patience=10))

    assert trainer._update_early_stopping({"val/mAP": 0.8}) is True
    assert trainer.early_stop_best_metric == pytest.approx(0.8)
    assert trainer._patience_counter == 0


def test_min_optimizer_step_ignores_earlier_observations(tmp_path: Path) -> None:
    trainer = _trainer(
        tmp_path,
        early_stop=_policy(min_optimizer_step=3, patience=1),
    )

    trainer.optimizer_step = 2
    assert trainer._update_early_stopping({"val/mAP": 0.9}) is False
    assert trainer.early_stop_best_metric == float("-inf")
    assert trainer._patience_counter == 0

    trainer.optimizer_step = 3
    assert trainer._update_early_stopping({"val/mAP": 0.4}) is False
    assert trainer.early_stop_best_metric == pytest.approx(0.4)
    assert trainer._patience_counter == 0


def test_enabled_policy_requires_finite_monitored_metric(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, early_stop=_policy())

    with pytest.raises(KeyError, match="monitor is missing"):
        trainer._update_early_stopping({"val/segm_AP": 0.5})
    with pytest.raises(ValueError, match="must be finite"):
        trainer._update_early_stopping({"val/mAP": float("nan")})
    with pytest.raises(ValueError, match="must be finite"):
        trainer._update_early_stopping({"val/mAP": float("inf")})


def test_best_selection_updates_early_stop_state_without_saving_training_state(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = _trainer(tmp_path, early_stop=_policy())

    def _unexpected_training_checkpoint(*args, **kwargs):
        del args, kwargs
        pytest.fail("best selection must not save a full training checkpoint")

    monkeypatch.setattr(trainer, "save_checkpoint", _unexpected_training_checkpoint)
    result = EvaluationResult(
        log_dict={"val/mAP": 0.5},
        coco_metrics={},
        coco_results_path=None,
        visualization_batch=None,
        visualization_outputs=None,
    )

    trainer._finalize_eval_result(result)

    assert trainer.best_metric == pytest.approx(0.5)
    assert trainer.early_stop_best_metric == pytest.approx(0.5)
    assert trainer._patience_counter == 0


@pytest.mark.parametrize("invalid_map", [float("nan"), float("inf")])
@pytest.mark.parametrize("policy", [{"enabled": False}, _policy(monitor="val/segm_AP")])
def test_non_finite_map_is_rejected_before_disabled_or_custom_monitor_state(
    tmp_path: Path, invalid_map: float, policy
) -> None:
    trainer = _trainer(tmp_path, early_stop=policy)
    result = EvaluationResult(
        log_dict={"val/mAP": invalid_map, "val/segm_AP": 0.5},
        coco_metrics={},
        coco_results_path=None,
        visualization_batch=None,
        visualization_outputs=None,
    )

    with pytest.raises(ValueError, match="val/mAP must be finite"):
        trainer._finalize_eval_result(result)

    assert trainer.best_metric == float("-inf")
    assert trainer.early_stop_best_metric == float("-inf")
    assert trainer._patience_counter == 0


def test_checkpoint_v3_round_trips_early_stop_state(tmp_path: Path) -> None:
    policy = _policy(patience=3)
    source = _trainer(tmp_path / "source", early_stop=policy)
    source.current_iter = 3
    source.optimizer_step = 3
    assert source._update_early_stopping({"val/mAP": 0.50}) is False
    assert source._update_early_stopping({"val/mAP": 0.505}) is False
    source.save_checkpoint()

    checkpoint = tmp_path / "source" / "checkpoint_iter_0000003.pth"
    resumed = _trainer(
        tmp_path / "resumed",
        early_stop=policy,
        resume=str(checkpoint),
    )

    assert resumed.current_iter == 3
    assert resumed.optimizer_step == 3
    assert resumed.early_stop_best_metric == pytest.approx(0.50)
    assert resumed._patience_counter == 1
    assert resumed.early_stop is False


@pytest.mark.parametrize("invalid_best", [float("nan"), float("inf")])
def test_malformed_v3_rejects_non_finite_best_metric(
    tmp_path: Path, invalid_best: float
) -> None:
    source = _trainer(tmp_path / "source")
    source.save_checkpoint()
    checkpoint_path = tmp_path / "source" / "checkpoint_iter_0000000.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["best_metric"] = invalid_best
    malformed_path = tmp_path / "malformed-best.pth"
    torch.save(checkpoint, malformed_path)

    with pytest.raises(RuntimeError, match="best_metric must be finite"):
        _trainer(tmp_path / "rejected", resume=str(malformed_path))


@pytest.mark.parametrize("invalid_best", [float("nan"), float("inf")])
def test_malformed_v3_rejects_non_finite_early_stop_best(
    tmp_path: Path, invalid_best: float
) -> None:
    policy = _policy()
    source = _trainer(tmp_path / "source", early_stop=policy)
    source.save_checkpoint()
    checkpoint_path = tmp_path / "source" / "checkpoint_iter_0000000.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["early_stop_state"]["early_stop_best_metric"] = invalid_best
    malformed_path = tmp_path / "malformed-early-best.pth"
    torch.save(checkpoint, malformed_path)

    with pytest.raises(RuntimeError, match="early_stop_best_metric must be finite"):
        _trainer(
            tmp_path / "rejected",
            early_stop=policy,
            resume=str(malformed_path),
        )


def test_malformed_v3_rejects_inconsistent_disabled_early_stop_state(
    tmp_path: Path,
) -> None:
    source = _trainer(tmp_path / "source")
    source.save_checkpoint()
    checkpoint_path = tmp_path / "source" / "checkpoint_iter_0000000.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["early_stop_state"]["early_stop_best_metric"] = 0.5
    malformed_path = tmp_path / "malformed-disabled-state.pth"
    torch.save(checkpoint, malformed_path)

    with pytest.raises(RuntimeError, match="disabled early-stop state"):
        _trainer(tmp_path / "rejected", resume=str(malformed_path))


def test_ddp_worker_receives_rank_zero_early_stop_state(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = _trainer(tmp_path, early_stop=_policy(patience=2))
    trainer.distributed = True
    trainer.rank = 1
    trainer.early_stop = False

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    rank_zero_state = trainer._early_stop_state_dict()
    rank_zero_state.update(
        early_stop_best_metric=0.7,
        patience_counter=2,
        should_stop=True,
    )

    def _broadcast(payload, src, device):
        assert src == 0
        assert device.type == "cpu"
        assert payload == [None]
        payload[0] = {"error": None, "state": rank_zero_state}

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", _broadcast)

    trainer._sync_early_stop_state()

    assert trainer.early_stop is True
    assert trainer.early_stop_best_metric == pytest.approx(0.7)
    assert trainer._patience_counter == 2


def test_ddp_rank_zero_finalization_error_is_synchronized(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = _trainer(tmp_path)
    trainer.distributed = True
    trainer.rank = 1

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def _broadcast(payload, src, device):
        assert src == 0
        assert device.type == "cpu"
        payload[0] = {
            "error": "KeyError: monitor missing",
            "state": trainer._early_stop_state_dict(),
        }

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", _broadcast)

    with pytest.raises(RuntimeError, match="monitor missing"):
        trainer._sync_early_stop_state()
