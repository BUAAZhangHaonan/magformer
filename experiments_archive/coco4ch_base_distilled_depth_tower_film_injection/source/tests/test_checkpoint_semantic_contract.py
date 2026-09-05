from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from magformer.engine.trainer import Trainer


class _Sampler:
    seed = 17


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
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state_dict) -> None:
        self.value = int(state_dict["value"])

    def step(self) -> None:
        pass


class _SchedulerA(_StatefulComponent):
    pass


class _SchedulerB(_StatefulComponent):
    pass


class _ScalerA(_StatefulComponent):
    pass


class _ScalerB(_StatefulComponent):
    pass


class _EmaA(_StatefulComponent):
    def __init__(self, *, decay: float = 0.8, warmup_iters: int = 2) -> None:
        super().__init__()
        self.decay = decay
        self.warmup_iters = warmup_iters


class _EmaB(_EmaA):
    pass


def _config(
    *,
    output_dir: str = "/volatile/output-a",
    resume: str | None = None,
    logger_dir: str = "/volatile/log-a",
    project: str = "project-a",
    model_variant: str = "tiny-a",
    data_variant: str = "data-a",
    base_lr: float = 0.1,
    gamma: float = 0.5,
    clip_value: float = 1.0,
    eval_period: int = 2,
):
    return {
        "model": {
            "meta_architecture": "Tiny",
            "variant": model_variant,
            "weights": "/semantic/init.pth",
        },
        "data": {
            "dataset_root": "/semantic/dataset",
            "variant": data_variant,
        },
        "solver": {
            "optimizer": "SGD",
            "base_lr": base_lr,
            "weight_decay": 0.0,
            "lr_scheduler": "step",
            "max_iter": 4,
            "warmup_factor": 0.1,
            "warmup_iters": 0,
            "warmup_method": "linear",
            "steps": [2],
            "gamma": gamma,
            "clip_value": clip_value,
        },
        "runtime": {
            "grad_accum_steps": 1,
            "early_stop": {"enabled": False},
            "seed": 11,
            "eval_period": eval_period,
            "checkpoint_period": 2,
            "output_dir": output_dir,
            "resume": resume,
            "logger": {
                "type": "tensorboard",
                "log_dir": logger_dir,
                "project": project,
                "entity": None,
                "run_name": "volatile-run",
            },
        },
        "vc_suda": {},
    }


def _trainer(
    tmp_path: Path,
    *,
    config=None,
    optimizer_cls=torch.optim.SGD,
    scheduler=True,
    scheduler_cls=_SchedulerA,
    amp=True,
    scaler_cls=_ScalerA,
    ema=True,
    ema_cls=_EmaA,
    ema_decay: float = 0.8,
    ema_warmup_iters: int = 2,
) -> Trainer:
    config = copy.deepcopy(_config() if config is None else config)
    model = torch.nn.Linear(1, 1)
    base_lr = float(config["solver"]["base_lr"])
    if optimizer_cls is torch.optim.SGD:
        optimizer = optimizer_cls(model.parameters(), lr=base_lr, momentum=0.9)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=base_lr)
    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        lr_scheduler=scheduler_cls() if scheduler else None,
        train_loader=_Loader(),
        config=config,
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=4,
        eval_period=int(config["runtime"]["eval_period"]),
        checkpoint_period=2,
        log_period=100,
        amp_enabled=False,
        clip_gradients=False,
        logger_config={"type": "none"},
    )
    trainer.amp_enabled = amp
    trainer.scaler = scaler_cls() if amp else None
    trainer.ema = ema_cls(decay=ema_decay, warmup_iters=ema_warmup_iters) if ema else None
    return trainer


def _checkpoint(source: Trainer, tmp_path: Path):
    source.save_checkpoint()
    path = tmp_path / "checkpoint_iter_0000000.pth"
    return torch.load(path, map_location="cpu", weights_only=True)


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
    destination: Trainer,
    checkpoint,
    tmp_path: Path,
    *,
    match: str,
) -> None:
    path = tmp_path / "candidate.pth"
    torch.save(checkpoint, path)
    before = {
        "model": copy.deepcopy(destination.model.state_dict()),
        "optimizer": copy.deepcopy(destination.optimizer.state_dict()),
        "rank": destination._capture_rank_state(),
    }
    with pytest.raises(RuntimeError, match=match):
        destination.resume(str(path))
    after = {
        "model": destination.model.state_dict(),
        "optimizer": destination.optimizer.state_dict(),
        "rank": destination._capture_rank_state(),
    }
    _assert_nested_equal(after, before)


@pytest.mark.parametrize("component", ["scheduler", "amp", "ema"])
@pytest.mark.parametrize("source_enabled", [True, False], ids=["on-to-off", "off-to-on"])
def test_component_enablement_mismatch_is_rejected_bidirectionally(
    tmp_path: Path, component: str, source_enabled: bool
) -> None:
    source_kwargs = {component: source_enabled}
    destination_kwargs = {component: not source_enabled}
    source = _trainer(tmp_path / "source", **source_kwargs)
    checkpoint = _checkpoint(source, tmp_path / "source")
    destination = _trainer(tmp_path / "destination", **destination_kwargs)
    _reject_without_mutation(destination, checkpoint, tmp_path, match="components")


@pytest.mark.parametrize("component", ["optimizer", "scheduler", "scaler", "ema"])
def test_component_class_mismatch_is_rejected(tmp_path: Path, component: str) -> None:
    source = _trainer(tmp_path / "source")
    checkpoint = _checkpoint(source, tmp_path / "source")
    kwargs = {
        "optimizer": {"optimizer_cls": torch.optim.Adam},
        "scheduler": {"scheduler_cls": _SchedulerB},
        "scaler": {"scaler_cls": _ScalerB},
        "ema": {"ema_cls": _EmaB},
    }[component]
    destination = _trainer(tmp_path / "destination", **kwargs)
    _reject_without_mutation(destination, checkpoint, tmp_path, match="components")


@pytest.mark.parametrize("policy", ["optimizer", "scheduler", "ema_decay", "ema_warmup"])
def test_component_policy_mismatch_is_rejected(tmp_path: Path, policy: str) -> None:
    source = _trainer(tmp_path / "source")
    checkpoint = _checkpoint(source, tmp_path / "source")
    destination_config = _config()
    kwargs = {}
    if policy == "optimizer":
        destination_config["solver"]["base_lr"] = 0.2
    elif policy == "scheduler":
        destination_config["solver"]["gamma"] = 0.7
    elif policy == "ema_decay":
        kwargs["ema_decay"] = 0.9
    else:
        kwargs["ema_warmup_iters"] = 3
    destination = _trainer(
        tmp_path / "destination",
        config=destination_config,
        **kwargs,
    )
    _reject_without_mutation(destination, checkpoint, tmp_path, match="components")


@pytest.mark.parametrize(
    "state_key",
    ["lr_scheduler_state_dict", "scaler_state_dict", "ema_state_dict"],
)
def test_disabled_component_rejects_unexpected_checkpoint_state(
    tmp_path: Path, state_key: str
) -> None:
    source = _trainer(
        tmp_path / "source",
        scheduler=False,
        amp=False,
        ema=False,
    )
    checkpoint = _checkpoint(source, tmp_path / "source")
    checkpoint[state_key] = {"unexpected": True}
    destination = _trainer(
        tmp_path / "destination",
        scheduler=False,
        amp=False,
        ema=False,
    )
    _reject_without_mutation(destination, checkpoint, tmp_path, match="unexpected")


def test_volatile_output_resume_and_log_destinations_may_change(tmp_path: Path) -> None:
    source = _trainer(
        tmp_path / "source",
        config=_config(
            output_dir="/volatile/source-output",
            resume=None,
            logger_dir="/volatile/source-logs",
            project="source-project",
        ),
    )
    checkpoint = _checkpoint(source, tmp_path / "source")
    path = tmp_path / "source" / "checkpoint_iter_0000000.pth"
    destination = _trainer(
        tmp_path / "destination",
        config=_config(
            output_dir="/volatile/destination-output",
            resume=str(path),
            logger_dir="/volatile/destination-logs",
            project="destination-project",
        ),
    )

    destination.resume(str(path))

    assert destination.current_iter == 0
    assert destination._build_resume_contract() == checkpoint["resume_contract"]
