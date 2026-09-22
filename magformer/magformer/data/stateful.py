"""Stateful data-loading primitives used by exact training resume."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, cast

import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import (
    StatefulDataLoader as TorchdataStatefulDataLoader,
)
from torchdata.stateful_dataloader.sampler import (
    RandomSampler as StatefulRandomSampler,
    StatefulDistributedSampler,
)

__all__ = [
    "EpochStatefulDistributedSampler",
    "ExactStatefulDataLoader",
    "LoaderStatePosition",
    "StatefulRandomSampler",
]


@dataclass(frozen=True)
class LoaderStatePosition:
    """Logical position described by an ExactStatefulDataLoader state dict."""

    yielded_batches: int
    total_batches: int
    iterator_finished: bool
    sampler_epoch: Optional[int]
    sampler_cursor_canonical: bool

    @property
    def exhausted(self) -> bool:
        return self.yielded_batches == self.total_batches

    @property
    def canonical(self) -> bool:
        return not self.iterator_finished and not self.exhausted and self.sampler_cursor_canonical


class ExactStatefulDataLoader(TorchdataStatefulDataLoader):
    """StatefulDataLoader that also owns its worker-seeding generator state.

    TorchData restores the current iterator's base seed and worker snapshots,
    but it does not restore ``DataLoader.generator`` for the next iterator.
    Keeping that generator independent from the sampler and persisting it here
    makes epoch-boundary continuation exact as well as mid-epoch continuation.
    """

    _GENERATOR_STATE = "_magformer_worker_generator_state"

    def __init__(self, *args, **kwargs) -> None:
        generator = kwargs.get("generator")
        if not isinstance(generator, torch.Generator):
            raise TypeError("ExactStatefulDataLoader requires an explicit torch.Generator")
        super().__init__(*args, **kwargs)
        self._pending_generator_state: Optional[torch.Tensor] = None
        self._pending_logical_state: Optional[Dict[str, Any]] = None
        self._canonical_logical_state: Optional[Dict[str, Any]] = None

    @staticmethod
    def _require_state_mapping(state: Mapping[str, Any], field: str) -> Mapping[str, Any]:
        if field not in state or not isinstance(state[field], Mapping):
            raise ValueError(f"DataLoader state {field} must be a mapping")
        return state[field]

    @staticmethod
    def _require_state_int(state: Mapping[str, Any], field: str) -> int:
        if field not in state or type(state[field]) is not int:
            raise ValueError(f"DataLoader state {field} must have type int")
        return state[field]

    @staticmethod
    def _validate_generator_state(state_dict: Mapping[str, Any]) -> torch.Tensor:
        if ExactStatefulDataLoader._GENERATOR_STATE not in state_dict:
            raise ValueError(
                "DataLoader state is missing " f"{ExactStatefulDataLoader._GENERATOR_STATE}"
            )
        generator_state = state_dict[ExactStatefulDataLoader._GENERATOR_STATE]
        if (
            not torch.is_tensor(generator_state)
            or generator_state.dtype != torch.uint8
            or generator_state.ndim != 1
        ):
            raise TypeError(
                "DataLoader worker generator state must be a one-dimensional " "uint8 tensor"
            )
        return generator_state

    def _state_from_live_iterator(self) -> Dict[str, Any]:
        state = dict(super().state_dict())
        if self._GENERATOR_STATE in state:
            raise RuntimeError(f"TorchData state unexpectedly contains {self._GENERATOR_STATE}")
        state[self._GENERATOR_STATE] = self.generator.get_state().clone()
        return state

    def _position_fields(
        self, state_dict: Mapping[str, Any]
    ) -> tuple[int, bool, Mapping[str, Any]]:
        if "_iterator_finished" not in state_dict:
            raise ValueError("DataLoader state is missing _iterator_finished")
        iterator_finished = state_dict["_iterator_finished"]
        if type(iterator_finished) is not bool:
            raise TypeError("DataLoader state _iterator_finished must have type bool")

        if self.num_workers == 0:
            yielded_batches = self._require_state_int(state_dict, "_num_yielded")
            index_sampler_state = self._require_state_mapping(state_dict, "_index_sampler_state")
        else:
            snapshot = self._require_state_mapping(state_dict, "_snapshot")
            snapshot_step = self._require_state_int(snapshot, "_snapshot_step")
            steps_since_snapshot = self._require_state_int(state_dict, "_steps_since_snapshot")
            yielded_batches = snapshot_step + steps_since_snapshot
            main_snapshot = self._require_state_mapping(snapshot, "_main_snapshot")
            index_sampler_state = self._require_state_mapping(main_snapshot, "_index_sampler_state")
        return yielded_batches, iterator_finished, index_sampler_state

    def inspect_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        expected_data_epoch: Optional[int] = None,
        expected_rank: Optional[int] = None,
        require_canonical: bool = False,
    ) -> LoaderStatePosition:
        """Purely validate and describe a serialized loader position.

        TorchData has separate single- and multi-worker state layouts.  This
        method is the only MAGFormer entry point that interprets those private
        layouts; Trainer code consumes the returned logical position only.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError("DataLoader state must be a mapping")
        self._validate_generator_state(state_dict)
        yielded_batches, iterator_finished, index_sampler_state = self._position_fields(state_dict)
        total_batches = len(self)
        if type(total_batches) is not int or total_batches < 0:
            raise RuntimeError(f"Exact resume requires a finite batch count, got {total_batches!r}")
        if yielded_batches < 0 or yielded_batches > total_batches:
            raise ValueError(
                "DataLoader yielded-batch position is outside the current epoch: "
                f"yielded={yielded_batches}, total={total_batches}"
            )

        sampler_epoch = None
        sampler_state = index_sampler_state.get("sampler_state")
        if isinstance(self.sampler, EpochStatefulDistributedSampler):
            if not isinstance(sampler_state, Mapping):
                raise ValueError("Distributed DataLoader state is missing sampler_state")
            sampler_epoch = _require_exact_int(sampler_state, "epoch")
            sampler_rank = _require_exact_int(sampler_state, "rank")
            if expected_data_epoch is not None and sampler_epoch != expected_data_epoch:
                raise ValueError(
                    "DataLoader sampler epoch does not match rank data_epoch: "
                    f"sampler={sampler_epoch}, data_epoch={expected_data_epoch}"
                )
            if expected_rank is not None and sampler_rank != expected_rank:
                raise ValueError(
                    "DataLoader sampler rank does not match checkpoint rank: "
                    f"sampler={sampler_rank}, rank={expected_rank}"
                )

        sampler_cursor_canonical = True
        cursor_error = None
        if yielded_batches == 0:
            samples_yielded = self._require_state_int(index_sampler_state, "samples_yielded")
            if samples_yielded != 0:
                sampler_cursor_canonical = False
                cursor_error = f"samples_yielded={samples_yielded}"
            for nested_field in ("sampler_state", "sampler_iter_state"):
                nested = index_sampler_state.get(nested_field)
                if isinstance(nested, Mapping) and "yielded" in nested:
                    nested_yielded = self._require_state_int(nested, "yielded")
                    if nested_yielded != 0:
                        sampler_cursor_canonical = False
                        cursor_error = f"{nested_field}.yielded={nested_yielded}"

        position = LoaderStatePosition(
            yielded_batches=yielded_batches,
            total_batches=total_batches,
            iterator_finished=iterator_finished,
            sampler_epoch=sampler_epoch,
            sampler_cursor_canonical=sampler_cursor_canonical,
        )
        if require_canonical:
            if iterator_finished:
                raise ValueError(
                    "Noncanonical DataLoader checkpoint state: " "_iterator_finished=true"
                )
            if position.exhausted:
                raise ValueError(
                    "Noncanonical DataLoader checkpoint state: exhausted epoch "
                    "must be advanced to the next epoch cursor 0"
                )
            if not sampler_cursor_canonical:
                raise ValueError(
                    "Noncanonical DataLoader checkpoint state: epoch cursor 0 "
                    f"contains {cursor_error}"
                )
        return position

    def validate_resume_state(
        self,
        state_dict: Mapping[str, Any],
        *,
        expected_data_epoch: int,
        expected_rank: int,
    ) -> Dict[str, Any]:
        """Return a safe copy only when a v3 loader state is canonical."""
        self.inspect_state_dict(
            state_dict,
            expected_data_epoch=expected_data_epoch,
            expected_rank=expected_rank,
            require_canonical=True,
        )
        return copy.deepcopy(dict(state_dict))

    def is_epoch_exhausted(self) -> bool:
        """Report whether the current iterator yielded its final epoch batch."""
        position = self.inspect_state_dict(self.state_dict())
        return position.exhausted

    def _epoch_start_index_sampler_state(self) -> Dict[str, Any]:
        batch_sampler_state = copy.deepcopy(dict(self.batch_sampler.state_dict()))
        if "samples_yielded" not in batch_sampler_state:
            raise RuntimeError(
                "Exact resume requires a stateful BatchSampler samples_yielded field"
            )
        batch_sampler_state["samples_yielded"] = 0

        has_stateful_sampler = "sampler_state" in batch_sampler_state
        has_stateful_iterator = "sampler_iter_state" in batch_sampler_state
        if has_stateful_sampler:
            sampler_state = self.sampler.state_dict()
            if not isinstance(sampler_state, Mapping) or "yielded" not in sampler_state:
                raise RuntimeError(
                    "Exact resume requires sampler_state.yielded at epoch boundaries"
                )
            sampler_state = copy.deepcopy(dict(sampler_state))
            sampler_state["yielded"] = 0
            batch_sampler_state["sampler_state"] = sampler_state
        if has_stateful_iterator:
            sampler_iterator = iter(self.sampler)
            if not callable(getattr(sampler_iterator, "state_dict", None)):
                raise RuntimeError("Exact resume requires a stateful sampler iterator")
            sampler_iter_state = sampler_iterator.state_dict()
            if (
                not isinstance(sampler_iter_state, Mapping)
                or sampler_iter_state.get("yielded") != 0
            ):
                raise RuntimeError("New sampler iterator must start with yielded=0")
            batch_sampler_state["sampler_iter_state"] = copy.deepcopy(dict(sampler_iter_state))
        if not (has_stateful_sampler or has_stateful_iterator):
            raise RuntimeError("Exact resume requires a stateful sampler or sampler iterator")
        return batch_sampler_state

    def _replace_index_sampler_state(
        self, state_dict: Dict[str, Any], index_sampler_state: Mapping[str, Any]
    ) -> None:
        replacement = copy.deepcopy(dict(index_sampler_state))
        if self.num_workers == 0:
            state_dict["_index_sampler_state"] = replacement
            return
        snapshot = cast(Dict[str, Any], self._require_state_mapping(state_dict, "_snapshot"))
        main_snapshot = cast(
            Dict[str, Any],
            self._require_state_mapping(snapshot, "_main_snapshot"),
        )
        main_snapshot["_index_sampler_state"] = replacement

    def start_next_epoch(self, epoch: int) -> Iterator[Any]:
        """Create the next iterator and expose a canonical cursor-0 state."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError(f"Next data epoch must be non-negative, got {epoch!r}")
        if self._pending_logical_state is not None:
            raise RuntimeError("Cannot advance an unapplied restored loader state")
        if self._canonical_logical_state is not None:
            raise RuntimeError("Next epoch was already canonicalized")
        if not self.is_epoch_exhausted():
            raise RuntimeError("Cannot start the next epoch before loader exhaustion")

        set_epoch = getattr(self.sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        canonical_index_state = self._epoch_start_index_sampler_state()
        iterator = super().__iter__()
        logical_state = copy.deepcopy(self._state_from_live_iterator())
        self._replace_index_sampler_state(logical_state, canonical_index_state)
        self.inspect_state_dict(
            logical_state,
            expected_data_epoch=epoch,
            expected_rank=getattr(self.sampler, "rank", None),
            require_canonical=True,
        )
        self._canonical_logical_state = logical_state
        return iterator

    def state_dict(self) -> Dict[str, Any]:
        if self._pending_logical_state is not None:
            return copy.deepcopy(self._pending_logical_state)
        live_state = self._state_from_live_iterator()
        if self._canonical_logical_state is not None:
            live_position = self.inspect_state_dict(live_state)
            if live_position.yielded_batches == 0 and not live_position.iterator_finished:
                return copy.deepcopy(self._canonical_logical_state)
            self._canonical_logical_state = None
        return live_state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("DataLoader state must be a mapping")
        generator_state = self._validate_generator_state(state_dict)
        logical_state = copy.deepcopy(dict(state_dict))
        torchdata_state = dict(logical_state)
        del torchdata_state[self._GENERATOR_STATE]
        super().load_state_dict(torchdata_state)
        self._canonical_logical_state = None
        self._pending_generator_state = generator_state.detach().cpu().clone()
        self._pending_logical_state = logical_state
        self.generator.set_state(self._pending_generator_state)

    def __iter__(self):
        pending_generator_state = self._pending_generator_state
        try:
            iterator = super().__iter__()
        except BaseException:
            if pending_generator_state is not None:
                self.generator.set_state(pending_generator_state)
            raise
        if pending_generator_state is not None:
            # Building the restored iterator consumes one provisional base seed.
            # Workers already received the checkpoint's saved base seed, so put
            # the independent generator back at the exact checkpoint position.
            self.generator.set_state(pending_generator_state)
            self._pending_generator_state = None
            self._pending_logical_state = None
        return iterator


def _require_exact_int(state: Mapping[str, Any], field: str) -> int:
    if field not in state:
        raise ValueError(f"Sampler state is missing {field}")
    value = state[field]
    if type(value) is not int:
        raise TypeError(f"Sampler state {field} must have type int, got {type(value).__name__}")
    return value


class EpochStatefulDistributedSampler(StatefulDistributedSampler):
    """StatefulDistributedSampler that also persists its permutation epoch.

    TorchData's sampler persists the number of yielded indices, but the
    upstream implementation intentionally leaves ``epoch`` to the caller.
    Exact mid-epoch resume needs both values, so this subclass stores the epoch
    and validates every static sharding input before applying saved progress.
    """

    _STATIC_FIELDS = (
        "num_replicas",
        "rank",
        "seed",
        "dataset_size",
        "num_samples",
        "total_size",
    )

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def state_dict(self) -> Dict[str, Any]:
        state = dict(super().state_dict())
        state.update(
            {
                "epoch": int(self.epoch),
                "num_replicas": int(self.num_replicas),
                "rank": int(self.rank),
                "seed": int(self.seed),
                "shuffle": bool(self.shuffle),
                "drop_last": bool(self.drop_last),
                "dataset_size": int(len(self.dataset)),
                "num_samples": int(self.num_samples),
                "total_size": int(self.total_size),
            }
        )
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("Sampler state must be a mapping, got " f"{type(state_dict).__name__}")

        epoch = _require_exact_int(state_dict, "epoch")
        if epoch < 0:
            raise ValueError(f"Sampler state epoch must be non-negative, got {epoch}")
        yielded = _require_exact_int(state_dict, "yielded")

        expected = {
            "num_replicas": int(self.num_replicas),
            "rank": int(self.rank),
            "seed": int(self.seed),
            "dataset_size": int(len(self.dataset)),
            "num_samples": int(self.num_samples),
            "total_size": int(self.total_size),
        }
        for field in self._STATIC_FIELDS:
            restored = _require_exact_int(state_dict, field)
            if restored != expected[field]:
                raise ValueError(
                    f"Sampler state {field} does not match the current loader: "
                    f"checkpoint={restored}, current={expected[field]}"
                )

        if yielded < 0:
            raise ValueError(f"Sampler state yielded must be non-negative, got {yielded}")
        if yielded > self.num_samples:
            raise ValueError(
                "Sampler state yielded exceeds this rank's sample count: "
                f"yielded={yielded}, num_samples={self.num_samples}"
            )

        for field, current in (
            ("shuffle", bool(self.shuffle)),
            ("drop_last", bool(self.drop_last)),
        ):
            if field not in state_dict or type(state_dict[field]) is not bool:
                raise TypeError(f"Sampler state {field} must have type bool")
            if state_dict[field] != current:
                raise ValueError(
                    f"Sampler state {field} does not match the current loader: "
                    f"checkpoint={state_dict[field]}, current={current}"
                )

        self.set_epoch(epoch)
        super().load_state_dict(dict(state_dict))
