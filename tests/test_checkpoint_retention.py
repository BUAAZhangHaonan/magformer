from types import SimpleNamespace

import torch

from magformer.config.schema import RuntimeConfig
from magformer.engine.trainer import Trainer
import tools.train as train_tool


def _make_trainer(tmp_path, runtime_config=None):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = {"runtime": runtime_config or {}}
    return Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        train_loader=[],
        val_loader=None,
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=0,
        eval_period=100,
        checkpoint_period=100,
        log_period=1,
        amp_enabled=False,
        config=config,
    )


def _write_numbered_checkpoints(tmp_path, *iters):
    for iteration in iters:
        (tmp_path / f"checkpoint_iter_{iteration:07d}.pth").write_text("checkpoint\n", encoding="utf-8")


def _checkpoint_names(tmp_path):
    return sorted(path.name for path in tmp_path.glob("checkpoint_iter_*.pth"))


def test_cleanup_old_checkpoints_keeps_two_by_default(tmp_path):
    trainer = _make_trainer(tmp_path)
    _write_numbered_checkpoints(tmp_path, 1, 2, 3, 4)

    trainer._cleanup_old_checkpoints()

    assert _checkpoint_names(tmp_path) == [
        "checkpoint_iter_0000003.pth",
        "checkpoint_iter_0000004.pth",
    ]


def test_cleanup_old_checkpoints_keeps_all_when_max_keep_is_none(tmp_path):
    trainer = _make_trainer(tmp_path)
    _write_numbered_checkpoints(tmp_path, 1, 2, 3, 4)

    trainer._cleanup_old_checkpoints(max_keep=None)

    assert _checkpoint_names(tmp_path) == [
        "checkpoint_iter_0000001.pth",
        "checkpoint_iter_0000002.pth",
        "checkpoint_iter_0000003.pth",
        "checkpoint_iter_0000004.pth",
    ]


def test_cleanup_old_checkpoints_keeps_all_when_max_keep_is_zero(tmp_path):
    trainer = _make_trainer(tmp_path)
    _write_numbered_checkpoints(tmp_path, 1, 2, 3, 4)

    trainer._cleanup_old_checkpoints(max_keep=0)

    assert _checkpoint_names(tmp_path) == [
        "checkpoint_iter_0000001.pth",
        "checkpoint_iter_0000002.pth",
        "checkpoint_iter_0000003.pth",
        "checkpoint_iter_0000004.pth",
    ]


def test_runtime_config_defaults_checkpoint_max_keep_to_two_and_allows_none():
    assert RuntimeConfig().checkpoint_max_keep == 2
    assert RuntimeConfig(checkpoint_max_keep=None).checkpoint_max_keep is None
    assert RuntimeConfig(checkpoint_max_keep=0).checkpoint_max_keep == 0


def test_trainer_reads_checkpoint_max_keep_from_runtime_config(tmp_path):
    trainer = _make_trainer(tmp_path, runtime_config={"checkpoint_max_keep": None})

    assert trainer.checkpoint_max_keep is None


def test_build_trainer_passes_checkpoint_max_keep(monkeypatch, tmp_path):
    captured = {}

    class DummyTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(train_tool, "Trainer", DummyTrainer)

    config = SimpleNamespace(
        runtime=SimpleNamespace(
            log_period=1,
            eval_period=10,
            checkpoint_period=10,
            resume=None,
            logger={},
            checkpoint_max_keep=None,
        ),
        solver=SimpleNamespace(max_iter=1, clip_gradients=True, clip_value=1.0),
        vc_suda=SimpleNamespace(enabled=False),
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_tool.build_trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        train_loader=[],
        val_loader=None,
        val_dataset=None,
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        is_distributed=False,
        amp_enabled=False,
    )

    assert captured["checkpoint_max_keep"] is None
