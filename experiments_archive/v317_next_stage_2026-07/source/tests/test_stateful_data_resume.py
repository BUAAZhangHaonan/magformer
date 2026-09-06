from __future__ import annotations

import copy
import gc
import random
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from magformer.data.dataset import CocoRgbdDataset, _InstanceBank
from magformer.data.semi_supervised_dataset import SemiSupervisedDataset
from magformer.data.stateful import (
    EpochStatefulDistributedSampler,
    ExactStatefulDataLoader,
    StatefulRandomSampler,
)
from tools.train import build_data_loaders


class _HistoryDataset(CocoRgbdDataset):
    """Small map dataset whose output depends on worker RNG and bank history."""

    def __init__(self, length: int = 8) -> None:
        self.length = length
        self._instance_bank = _InstanceBank(capacity=6, small_threshold=64)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
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
        return torch.tensor(
            [
                index,
                sampled[0]["label"],
                sampled[1]["label"],
                random.randint(0, 2**30),
                int(np.random.randint(0, 2**30)),
                int(torch.randint(0, 2**30, ()).item()),
                len(self._instance_bank),
            ],
            dtype=torch.int64,
        )


class _PlainDataset(Dataset):
    def __init__(self) -> None:
        self.transform = None

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        raise AssertionError("Loader construction must not fetch samples")


class _IndexDataset(Dataset):
    transform = None

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        return torch.tensor(index, dtype=torch.int64)


class _NestedStatefulDataset(Dataset):
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self):
        return {"value": int(self.value)}

    def load_state_dict(self, state) -> None:
        if not isinstance(state, Mapping) or type(state.get("value")) is not int:
            raise TypeError("Nested test dataset state is malformed")
        self.value = state["value"]


class _RepeatingBatchStream:
    def __init__(self, loader: ExactStatefulDataLoader) -> None:
        self.loader = loader
        self.iterator = iter(loader)

    def take(self, count: int) -> list[torch.Tensor]:
        batches = []
        while len(batches) < count:
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
                continue
            batches.append(batch.clone())
        return batches

    def close(self) -> None:
        shutdown = getattr(self.iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()


def _history_loader() -> ExactStatefulDataLoader:
    dataset = _HistoryDataset()
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(9127)
    worker_generator = torch.Generator()
    worker_generator.manual_seed(8123)
    return ExactStatefulDataLoader(
        dataset,
        batch_size=2,
        sampler=StatefulRandomSampler(dataset, generator=sampler_generator),
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=False,
        generator=worker_generator,
        snapshot_every_n_steps=1,
    )


def _index_loader(num_workers: int) -> ExactStatefulDataLoader:
    dataset = _IndexDataset()
    kwargs = {}
    if num_workers > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=False)
    return ExactStatefulDataLoader(
        dataset,
        batch_size=2,
        sampler=StatefulRandomSampler(
            dataset,
            generator=torch.Generator().manual_seed(9127),
        ),
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(8123),
        snapshot_every_n_steps=1,
        **kwargs,
    )


def _assert_batches_equal(actual: list[torch.Tensor], expected: list[torch.Tensor]) -> None:
    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected):
        assert torch.equal(actual_batch, expected_batch)


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


@pytest.mark.parametrize(
    "cut",
    [2, 4, 6],
    ids=["mid-epoch", "last-batch", "cross-epoch"],
)
def test_stateful_loader_restores_worker_rng_prefetch_shuffle_and_bank_history(
    tmp_path: Path, cut: int
) -> None:
    loader = _history_loader()
    uninterrupted = _RepeatingBatchStream(loader)
    uninterrupted.take(cut)

    checkpoint_path = tmp_path / f"loader-state-{cut}.pth"
    torch.save(loader.state_dict(), checkpoint_path)
    loader_state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    expected_tail = uninterrupted.take(5)
    uninterrupted.close()
    del uninterrupted, loader
    gc.collect()

    resumed_loader = _history_loader()
    resumed_loader.load_state_dict(loader_state)
    resumed = _RepeatingBatchStream(resumed_loader)
    actual_tail = resumed.take(5)
    resumed.close()

    _assert_batches_equal(actual_tail, expected_tail)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_epoch_boundary_api_canonicalizes_next_epoch_cursor_zero(
    num_workers: int,
) -> None:
    loader = _index_loader(num_workers)
    iterator = iter(loader)
    for batch_index in range(len(loader)):
        next(iterator)
        assert loader.is_epoch_exhausted() is (batch_index == len(loader) - 1)

    stale_boundary = loader.state_dict()
    with pytest.raises(ValueError, match="exhausted epoch"):
        loader.validate_resume_state(
            stale_boundary,
            expected_data_epoch=0,
            expected_rank=0,
        )

    next_iterator = loader.start_next_epoch(1)
    canonical_state = loader.validate_resume_state(
        loader.state_dict(),
        expected_data_epoch=1,
        expected_rank=0,
    )
    position = loader.inspect_state_dict(canonical_state, require_canonical=True)
    assert position.yielded_batches == 0
    assert position.iterator_finished is False

    expected = [next(next_iterator).clone(), next(next_iterator).clone()]
    restored_loader = _index_loader(num_workers)
    restored_loader.load_state_dict(canonical_state)
    restored_iterator = iter(restored_loader)
    actual = [next(restored_iterator).clone(), next(restored_iterator).clone()]
    _assert_batches_equal(actual, expected)

    for active_iterator in (next_iterator, restored_iterator):
        shutdown = getattr(active_iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resume_state_rejects_iterator_finished_epoch_boundary(
    num_workers: int,
) -> None:
    loader = _index_loader(num_workers)
    iterator = iter(loader)
    list(iterator)
    with pytest.raises(ValueError, match="_iterator_finished=true"):
        loader.validate_resume_state(
            loader.state_dict(),
            expected_data_epoch=0,
            expected_rank=0,
        )


def test_loaded_loader_state_is_idempotent_before_iter_and_exact_across_epochs() -> None:
    loader = _history_loader()
    uninterrupted = _RepeatingBatchStream(loader)
    uninterrupted.take(3)
    saved_state = copy.deepcopy(loader.state_dict())
    expected_tail = uninterrupted.take(12)
    uninterrupted.close()

    first_restore = _history_loader()
    first_restore.load_state_dict(saved_state)
    assert first_restore._iterator is None
    _assert_nested_equal(first_restore.state_dict(), saved_state)
    assert first_restore._iterator is None
    assert torch.equal(
        first_restore.generator.get_state(),
        saved_state[ExactStatefulDataLoader._GENERATOR_STATE],
    )

    round_tripped_state = first_restore.state_dict()
    second_restore = _history_loader()
    second_restore.load_state_dict(round_tripped_state)
    _assert_nested_equal(second_restore.state_dict(), saved_state)

    resumed = _RepeatingBatchStream(second_restore)
    actual_tail = resumed.take(12)
    resumed.close()

    _assert_batches_equal(actual_tail, expected_tail)


def test_distributed_sampler_restores_epoch_and_mid_epoch_cursor(
    tmp_path: Path,
) -> None:
    dataset = list(range(11))
    sampler = EpochStatefulDistributedSampler(
        dataset,
        num_replicas=2,
        rank=1,
        shuffle=True,
        seed=73,
        drop_last=False,
    )
    sampler.set_epoch(5)
    iterator = iter(sampler)
    prefix = [next(iterator), next(iterator)]
    assert len(prefix) == 2

    state_path = tmp_path / "sampler-state.pth"
    torch.save(sampler.state_dict(), state_path)
    expected_tail = list(iterator)

    restored = EpochStatefulDistributedSampler(
        dataset,
        num_replicas=2,
        rank=1,
        shuffle=True,
        seed=73,
        drop_last=False,
    )
    restored.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))

    assert restored.epoch == 5
    assert list(restored) == expected_tail


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"rank": 1}, "rank does not match"),
        ({"seed": 74}, "seed does not match"),
        ({"num_replicas": 3}, "num_replicas does not match"),
    ],
)
def test_distributed_sampler_rejects_different_sharding_contract(overrides, match: str) -> None:
    source = EpochStatefulDistributedSampler(
        list(range(11)),
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=73,
    )
    source.set_epoch(2)
    next(iter(source))

    current = {
        "num_replicas": 2,
        "rank": 0,
        "shuffle": True,
        "seed": 73,
    }
    current.update(overrides)
    incompatible = EpochStatefulDistributedSampler(
        list(range(11)),
        **current,
    )
    with pytest.raises(ValueError, match=match):
        incompatible.load_state_dict(source.state_dict())


@pytest.mark.parametrize(
    ("cursor", "exception_type", "match"),
    [
        (None, ValueError, "missing yielded"),
        (True, TypeError, "yielded must have type int"),
        (1.0, TypeError, "yielded must have type int"),
        ("1", TypeError, "yielded must have type int"),
        (-1, ValueError, "yielded must be non-negative"),
        (7, ValueError, "yielded exceeds"),
    ],
)
def test_distributed_sampler_rejects_malformed_cursor(cursor, exception_type, match: str) -> None:
    source = EpochStatefulDistributedSampler(
        list(range(11)),
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=73,
    )
    state = source.state_dict()
    if cursor is None:
        del state["yielded"]
    else:
        state["yielded"] = cursor

    destination = EpochStatefulDistributedSampler(
        list(range(11)),
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=73,
    )
    with pytest.raises(exception_type, match=match):
        destination.load_state_dict(state)


def test_semi_supervised_dataset_forwards_branch_state_and_restores_own_rng(
    tmp_path: Path,
) -> None:
    dataset = SemiSupervisedDataset.__new__(SemiSupervisedDataset)
    dataset.source = _NestedStatefulDataset(11)
    dataset.target_labeled = _NestedStatefulDataset(22)
    dataset.target_unlabeled = _NestedStatefulDataset(33)

    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    state_path = tmp_path / "semi-worker-state.pth"
    torch.save(dataset.state_dict(), state_path)
    expected_draws = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(()).item()),
    )

    dataset.source.value = -1
    dataset.target_labeled.value = -2
    dataset.target_unlabeled.value = -3
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    dataset.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))

    assert dataset.source.value == 11
    assert dataset.target_labeled.value == 22
    assert dataset.target_unlabeled.value == 33
    assert random.random() == expected_draws[0]
    assert float(np.random.random()) == expected_draws[1]
    assert float(torch.rand(()).item()) == expected_draws[2]


def test_train_builder_uses_stateful_loader_and_independent_generators() -> None:
    config = SimpleNamespace(
        runtime=SimpleNamespace(seed=41),
        data=SimpleNamespace(
            image_size=8,
            min_scale=1.0,
            max_scale=1.0,
            random_flip="none",
            rgb_photo_aug=SimpleNamespace(
                brightness=0.0,
                contrast=0.0,
                saturation=0.0,
                hue=0.0,
            ),
            depth=SimpleNamespace(
                scale=1.0,
                shift=0.0,
                clip_min=0.0,
                clip_max=1.0,
                norm="none",
                per_sample_norm=False,
            ),
            depth_noise=SimpleNamespace(
                gaussian_std=0.0,
                speckle_std=0.0,
                drop_prob=0.0,
                drop_val=0.0,
            ),
            copy_paste=SimpleNamespace(enabled=False),
            sahi_crop_size=None,
        ),
    )
    train_loader, val_loader = build_data_loaders(
        config,
        _PlainDataset(),
        _PlainDataset(),
        batch_size=2,
        num_workers=0,
        is_distributed=False,
    )

    assert isinstance(train_loader, ExactStatefulDataLoader)
    assert isinstance(train_loader, StatefulDataLoader)
    assert train_loader.generator is not val_loader.generator
    assert train_loader.sampler.generator.initial_seed() == 41
    assert train_loader.generator.initial_seed() == 1_000_044
    assert val_loader.generator.initial_seed() == 2_000_047
