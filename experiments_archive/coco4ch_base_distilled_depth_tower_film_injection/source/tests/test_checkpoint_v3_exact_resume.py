from __future__ import annotations

import copy
import gc
import random
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magformer.data.dataset import CocoRgbdDataset, _InstanceBank
from magformer.data.stateful import (
    EpochStatefulDistributedSampler,
    ExactStatefulDataLoader,
    StatefulRandomSampler,
)
from magformer.engine.trainer import DDPTrainer, Trainer


class _HistoryTrainingDataset(CocoRgbdDataset):
    """Map dataset driven by worker RNG and mutable copy-paste history."""

    def __init__(self, length: int = 12) -> None:
        self.length = length
        self.transform = None
        self._instance_bank = _InstanceBank(capacity=8, small_threshold=64)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        image = np.full((8, 8, 3), index, dtype=np.uint8)
        masks = np.zeros((8, 8, 1), dtype=bool)
        masks[1:5, 1:5, 0] = True
        self._instance_bank.deposit(
            image=image,
            masks=masks,
            boxes=np.array([[1, 1, 5, 5]], dtype=np.float32),
            labels=np.array([index], dtype=np.int64),
        )
        sampled = self._instance_bank.sample(2, prefer_small=True)
        python_draw = random.randint(0, 2**30)
        numpy_draw = int(np.random.randint(0, 2**30))
        torch_draw = int(torch.randint(0, 2**30, ()).item())
        features = torch.tensor(
            [
                float(index) / self.length,
                float(sampled[0]["label"]) / self.length,
                float(sampled[1]["label"]) / self.length,
                python_draw / float(2**30),
                numpy_draw / float(2**30),
                torch_draw / float(2**30),
                float(len(self._instance_bank)) / 8.0,
            ],
            dtype=torch.float32,
        )
        return {
            "sample_id": torch.tensor(index, dtype=torch.int64),
            "images": features,
            "depths": torch.tensor([features[0], features[-1]], dtype=torch.float32),
        }


class _TinyStochasticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rgb = nn.Linear(7, 1)
        self.depth = nn.Linear(2, 1)

    def forward(
        self,
        images,
        depths,
        targets=None,
        padding_masks=None,
        depth_noise_masks=None,
        depth_valid_masks=None,
    ):
        del targets, padding_masks, depth_noise_masks, depth_valid_masks
        main_rng_noise = (
            torch.rand((), device=images.device)
            + torch.tensor(random.random(), device=images.device)
            + torch.tensor(float(np.random.random()), device=images.device)
        ) * 1.0e-3
        prediction = self.rgb(images).mean() + self.depth(depths).mean()
        return {"total_loss": (prediction + main_rng_noise).square()}


class _RecordingTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        self.sample_ids = []
        self.batches = []
        self.losses = []
        super().__init__(*args, **kwargs)

    def _train_step(self, batch):
        self.sample_ids.append(tuple(int(value) for value in batch["sample_id"]))
        self.batches.append(_clone_state(batch))
        losses = super()._train_step(batch)
        self.losses.append(float(losses["total_loss"].detach().cpu()))
        return losses

    def evaluate(self):
        return {}


class _RecordingDDPTrainer(DDPTrainer, _RecordingTrainer):
    def evaluate(self):
        return {}


def _set_global_seed(seed: int = 31415) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _history_loader(
    *,
    rank: int = 0,
    world_size: int = 1,
    batch_size: int = 2,
) -> ExactStatefulDataLoader:
    dataset = _HistoryTrainingDataset()
    worker_generator = torch.Generator().manual_seed(8123 + rank)
    if world_size > 1:
        sampler = EpochStatefulDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=9127,
            drop_last=False,
        )
    else:
        sampler_generator = torch.Generator().manual_seed(9127)
        sampler = StatefulRandomSampler(dataset, generator=sampler_generator)
    return ExactStatefulDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=False,
        generator=worker_generator,
        snapshot_every_n_steps=1,
    )


def _make_trainer(
    output_dir: Path,
    *,
    max_iter: int,
    rank: int = 0,
    world_size: int = 1,
    resume: str | None = None,
    batch_size: int = 2,
):
    model = _TinyStochasticModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    trainer_type = _RecordingDDPTrainer if world_size > 1 else _RecordingTrainer
    kwargs = {}
    if world_size > 1:
        kwargs["find_unused_parameters"] = False
    return trainer_type(
        model=model,
        criterion=None,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=_history_loader(
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
        ),
        config={
            "solver": {"iteration_unit": "legacy_micro_step"},
            "runtime": {
                "grad_accum_steps": 2,
                "early_stop": {"enabled": False},
                "ema_enabled": True,
                "ema_decay": 0.8,
                "ema_warmup_iters": 2,
            },
            "data": {"test_pipeline": "history-v1"},
        },
        device=torch.device("cpu"),
        output_dir=str(output_dir),
        max_iter=max_iter,
        eval_period=6,
        checkpoint_period=6,
        log_period=100,
        amp_enabled=False,
        clip_gradients=False,
        resume=resume,
        logger_config={"type": "none"},
        **kwargs,
    )


def _clone_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return copy.deepcopy(value)


def _result(trainer) -> dict:
    loader_position = trainer.train_loader.inspect_state_dict(
        trainer.train_loader.state_dict(),
        expected_data_epoch=trainer.data_epoch,
        expected_rank=trainer.rank,
        require_canonical=True,
    )
    return {
        "sample_ids": list(trainer.sample_ids),
        "batches": list(trainer.batches),
        "losses": list(trainer.losses),
        "model": _clone_state(trainer._model_state_target().state_dict()),
        "optimizer": _clone_state(trainer.optimizer.state_dict()),
        "scheduler": _clone_state(trainer.lr_scheduler.state_dict()),
        "ema": _clone_state(trainer.ema.state_dict()),
        "counters": {
            "current_iter": trainer.current_iter,
            "optimizer_step": trainer.optimizer_step,
            "amp_skipped_steps": trainer.amp_skipped_steps,
            "consecutive_amp_skips": trainer.consecutive_amp_skips,
            "accum_count": trainer._accum_count,
            "data_epoch": trainer.data_epoch,
        },
        "loader_position": {
            "yielded_batches": loader_position.yielded_batches,
            "total_batches": loader_position.total_batches,
            "iterator_finished": loader_position.iterator_finished,
            "sampler_epoch": loader_position.sampler_epoch,
        },
        "next_rng": {
            "python": random.random(),
            "numpy": float(np.random.random()),
            "torch": torch.rand(4),
        },
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


def _assert_exact_run(uninterrupted: dict, prefix: dict, resumed: dict) -> None:
    assert prefix["sample_ids"] + resumed["sample_ids"] == uninterrupted["sample_ids"]
    _assert_nested_equal(
        prefix["batches"] + resumed["batches"], uninterrupted["batches"]
    )
    assert prefix["losses"] + resumed["losses"] == uninterrupted["losses"]
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "ema",
        "counters",
        "loader_position",
        "next_rng",
    ):
        _assert_nested_equal(resumed[key], uninterrupted[key])


def test_single_process_last_batch_checkpoint_resumes_exactly(tmp_path: Path) -> None:
    _set_global_seed()
    uninterrupted_trainer = _make_trainer(tmp_path / "uninterrupted", max_iter=12)
    uninterrupted_trainer.train()
    uninterrupted = _result(uninterrupted_trainer)

    _set_global_seed()
    prefix_trainer = _make_trainer(tmp_path / "split", max_iter=6)
    prefix_trainer.train()
    prefix = _result(prefix_trainer)
    checkpoint = tmp_path / "split" / "last.pt"
    assert prefix["counters"]["data_epoch"] == 1
    assert prefix["loader_position"]["yielded_batches"] == 0
    assert prefix["loader_position"]["iterator_finished"] is False
    del prefix_trainer
    gc.collect()

    _set_global_seed()
    resumed_trainer = _make_trainer(
        tmp_path / "resumed",
        max_iter=12,
        resume=str(checkpoint),
    )
    resumed_trainer.train()
    resumed = _result(resumed_trainer)

    _assert_exact_run(uninterrupted, prefix, resumed)
    del uninterrupted_trainer, resumed_trainer
    gc.collect()


class _EpochDataset(Dataset):
    transform = None

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int):
        value = torch.tensor([float(index)], dtype=torch.float32)
        return {
            "sample_id": torch.tensor(index),
            "images": value.repeat(7),
            "depths": value.repeat(2),
        }


class _RecordingEpochSampler(EpochStatefulDistributedSampler):
    def __init__(self, dataset) -> None:
        self.epoch_calls = []
        super().__init__(
            dataset,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=73,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch_calls.append(int(epoch))
        super().set_epoch(epoch)


def _epoch_loader():
    dataset = _EpochDataset()
    return ExactStatefulDataLoader(
        dataset,
        batch_size=1,
        sampler=_RecordingEpochSampler(dataset),
        num_workers=0,
        generator=torch.Generator().manual_seed(91),
        snapshot_every_n_steps=1,
    )


def _epoch_trainer(
    output_dir: Path,
    *,
    max_iter: int,
    resume: str | None = None,
):
    model = _TinyStochasticModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return _RecordingTrainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=_epoch_loader(),
        config={
            "solver": {"iteration_unit": "legacy_micro_step"},
            "runtime": {
                "grad_accum_steps": 1,
                "early_stop": {"enabled": False},
                "ema_enabled": True,
                "ema_decay": 0.8,
                "ema_warmup_iters": 1,
            },
            "data": {"test_pipeline": "epoch-v1"},
        },
        device=torch.device("cpu"),
        output_dir=str(output_dir),
        max_iter=max_iter,
        eval_period=2,
        checkpoint_period=2,
        log_period=100,
        amp_enabled=False,
        clip_gradients=False,
        resume=resume,
        logger_config={"type": "none"},
    )


def _begin_epoch_five(trainer) -> None:
    trainer.data_epoch = 5
    trainer.start_epoch = 5


def test_resume_mid_epoch_five_advances_to_epoch_six_exactly(tmp_path: Path) -> None:
    _set_global_seed()
    uninterrupted_trainer = _epoch_trainer(tmp_path / "epoch-uninterrupted", max_iter=7)
    _begin_epoch_five(uninterrupted_trainer)
    uninterrupted_trainer.train()
    uninterrupted = _result(uninterrupted_trainer)

    _set_global_seed()
    source = _epoch_trainer(tmp_path / "epoch-source", max_iter=2)
    _begin_epoch_five(source)
    source.train()
    prefix = _result(source)
    checkpoint = tmp_path / "epoch-source" / "last.pt"

    _set_global_seed()
    resumed = _epoch_trainer(
        tmp_path / "epoch-resumed",
        max_iter=7,
        resume=str(checkpoint),
    )
    calls_after_resume = list(resumed.train_loader.sampler.epoch_calls)
    # Loader restoration is intentionally lazy: resume itself creates no
    # iterator and therefore does not apply the saved sampler state yet.
    assert calls_after_resume == []
    resumed.train()

    assert resumed.data_epoch == 6
    assert resumed.start_epoch == 6
    assert resumed.train_loader.sampler.epoch_calls == [5, 6]
    _assert_exact_run(uninterrupted, prefix, _result(resumed))


def test_resume_contract_mismatch_rejects_before_mutation(tmp_path: Path) -> None:
    _set_global_seed()
    source = _make_trainer(tmp_path / "contract-source", max_iter=6)
    source.train()
    checkpoint = tmp_path / "contract-source" / "last.pt"

    _set_global_seed()
    destination = _make_trainer(
        tmp_path / "contract-destination",
        max_iter=12,
        batch_size=1,
    )
    model_before = _clone_state(destination.model.state_dict())
    loader_before = _clone_state(destination.train_loader.state_dict())
    rng_before = destination._capture_rank_state()

    with pytest.raises(RuntimeError, match="per_rank_batch_size"):
        destination.resume(str(checkpoint))

    _assert_nested_equal(destination.model.state_dict(), model_before)
    _assert_nested_equal(destination.train_loader.state_dict(), loader_before)
    _assert_nested_equal(destination._capture_rank_state(), rng_before)


def test_resume_rejects_noncanonical_iterator_finished_before_mutation(
    tmp_path: Path,
) -> None:
    _set_global_seed()
    source = _epoch_trainer(tmp_path / "finished-source", max_iter=2)
    _begin_epoch_five(source)
    source.train()
    checkpoint_path = tmp_path / "finished-source" / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["rank_states"][0]["train_loader_state_dict"]["_iterator_finished"] = True
    malformed_path = tmp_path / "last.pt"
    torch.save(checkpoint, malformed_path)

    _set_global_seed()
    destination = _epoch_trainer(tmp_path / "finished-destination", max_iter=7)
    model_before = _clone_state(destination.model.state_dict())
    loader_before = _clone_state(destination.train_loader.state_dict())
    rng_before = destination._capture_rank_state()

    with pytest.raises(ValueError, match="_iterator_finished=true"):
        destination.resume(str(malformed_path))

    _assert_nested_equal(destination.model.state_dict(), model_before)
    _assert_nested_equal(destination.train_loader.state_dict(), loader_before)
    _assert_nested_equal(destination._capture_rank_state(), rng_before)


def _distributed_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
    phase: str,
    max_iter: int,
    resume: str | None,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _set_global_seed()
        trainer = _make_trainer(
            Path(output_dir),
            max_iter=max_iter,
            rank=rank,
            world_size=world_size,
            resume=resume,
        )
        trainer.train()
        result = _result(trainer)
        torch.save(result, Path(output_dir) / f"{phase}_rank{rank}.pth")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_distributed_phase(
    tmp_path: Path,
    *,
    phase: str,
    output_dir: Path,
    max_iter: int,
    resume: Path | None = None,
) -> None:
    init_file = tmp_path / f"gloo-init-{phase}"
    mp.spawn(
        _distributed_worker,
        args=(
            2,
            str(init_file),
            str(output_dir),
            phase,
            max_iter,
            None if resume is None else str(resume),
        ),
        nprocs=2,
        join=True,
    )


def test_two_rank_gloo_last_batch_process_restart_is_exact(tmp_path: Path) -> None:
    uninterrupted_dir = tmp_path / "ddp-uninterrupted"
    split_dir = tmp_path / "ddp-split"
    resumed_dir = tmp_path / "ddp-resumed"
    _run_distributed_phase(
        tmp_path,
        phase="uninterrupted",
        output_dir=uninterrupted_dir,
        max_iter=12,
    )
    _run_distributed_phase(
        tmp_path,
        phase="prefix",
        output_dir=split_dir,
        max_iter=6,
    )
    checkpoint = split_dir / "last.pt"
    _run_distributed_phase(
        tmp_path,
        phase="resumed",
        output_dir=resumed_dir,
        max_iter=12,
        resume=checkpoint,
    )

    for rank in range(2):
        uninterrupted = torch.load(
            uninterrupted_dir / f"uninterrupted_rank{rank}.pth",
            map_location="cpu",
            weights_only=True,
        )
        prefix = torch.load(
            split_dir / f"prefix_rank{rank}.pth",
            map_location="cpu",
            weights_only=True,
        )
        resumed = torch.load(
            resumed_dir / f"resumed_rank{rank}.pth",
            map_location="cpu",
            weights_only=True,
        )
        assert prefix["counters"]["data_epoch"] == 2
        assert prefix["loader_position"]["yielded_batches"] == 0
        assert prefix["loader_position"]["iterator_finished"] is False
        assert prefix["loader_position"]["sampler_epoch"] == 2
        _assert_exact_run(uninterrupted, prefix, resumed)
