# -*- coding: utf-8 -*-
"""
Semi-Supervised Dataset for VC-SUDA

Combines labeled source data with unlabeled target data,
providing weak/strong augmentation pairs for teacher-student training.
"""

import copy
import random
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import (
    CocoRgbdDataset,
    _decode_numpy_rng_state,
    _decode_python_rng_state,
    _decode_torch_rng_state,
    _encode_numpy_rng_state,
    _encode_python_rng_state,
)


_LABEL_KEYS = ("labels", "masks", "boxes", "annotations")
_GEOMETRY_TRANSFORM_NAMES = {
    "InitContentMask",
    "RandomFlip",
    "ResizeScale",
    "FixedSizeCrop",
}
_WORKER_STATE_VERSION = 1


class SemiSupervisedDataset(Dataset):
    """
    Semi-supervised dataset that yields paired (weak_aug, strong_aug) samples
    for unlabeled target data, alongside standard labeled source samples.

    Returns dict with keys depending on stage:
    - Stage A/B: {"source": source_sample}
    - Stage C+: {"source": source_sample, "target_weak": weak_sample, "target_strong": strong_sample}
    """

    def __init__(
        self,
        # Source (labeled synthetic)
        source_root: str,
        source_ann: str,
        source_split: str = "train",
        source_transform: Optional[Any] = None,
        # Target labeled (small real subset, Stage B+)
        target_labeled_root: Optional[str] = None,
        target_labeled_ann: Optional[str] = None,
        target_labeled_split: str = "train",
        target_labeled_transform: Optional[Any] = None,
        # Target unlabeled (Stage C+)
        target_unlabeled_root: Optional[str] = None,
        target_unlabeled_ann: Optional[str] = None,
        target_unlabeled_split: str = "train",
        weak_transform: Optional[Any] = None,
        strong_transform: Optional[Any] = None,
        # Stage control
        stage: str = "A",
    ):
        super().__init__()

        self.stage = stage

        # Source dataset (always present, labeled, strong augmentation)
        self.source = CocoRgbdDataset(
            dataset_root=source_root,
            ann_file=source_ann,
            split=source_split,
            transform=source_transform,
            is_train=True,
            has_annotations=True,
        )

        # Target labeled dataset (Stage B+, small subset)
        self.target_labeled = None
        if target_labeled_root and target_labeled_ann and stage in ("B", "C", "D", "E"):
            self.target_labeled = CocoRgbdDataset(
                dataset_root=target_labeled_root,
                ann_file=target_labeled_ann,
                split=target_labeled_split,
                transform=target_labeled_transform or source_transform,
                is_train=True,
                has_annotations=True,
            )

        # Target unlabeled dataset (Stage C+, weak + strong pair)
        self.target_unlabeled = None
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform or source_transform

        if target_unlabeled_root and target_unlabeled_ann and stage in ("C", "D", "E"):
            # The manifest may contain GT annotations, but the unlabeled branch
            # treats it strictly as an image manifest. Labels are stripped and
            # asserted absent before samples leave this dataset.
            self._target_unlabeled_base = CocoRgbdDataset(
                dataset_root=target_unlabeled_root,
                ann_file=target_unlabeled_ann,
                split=target_unlabeled_split,
                transform=None,
                is_train=False,
                has_annotations=True,
            )
            self.target_unlabeled = self._target_unlabeled_base

        print(f"[SemiSupervisedDataset] Stage={stage}, "
              f"source={len(self.source)}, "
              f"target_labeled={len(self.target_labeled) if self.target_labeled else 0}, "
              f"target_unlabeled={len(self.target_unlabeled) if self.target_unlabeled else 0}")

    def __len__(self) -> int:
        # Length is determined by the larger of source and unlabeled
        lengths = [len(self.source)]
        if self.target_unlabeled is not None:
            lengths.append(len(self.target_unlabeled))
        return max(lengths)

    @staticmethod
    def _child_state_dict(dataset: Optional[Dataset]) -> Optional[Dict[str, Any]]:
        if dataset is None:
            return None
        state_dict = getattr(dataset, "state_dict", None)
        if not callable(state_dict):
            raise TypeError(
                f"Nested dataset {type(dataset).__name__} must define state_dict()"
            )
        state = state_dict()
        if not isinstance(state, Mapping):
            raise TypeError("Nested dataset state_dict() must return a mapping")
        return dict(state)

    def state_dict(self) -> Dict[str, Any]:
        """Return complete state for one semi-supervised worker replica."""
        return {
            "version": _WORKER_STATE_VERSION,
            "python_rng": _encode_python_rng_state(),
            "numpy_rng": _encode_numpy_rng_state(),
            "torch_rng": torch.get_rng_state().clone(),
            "source": self._child_state_dict(self.source),
            "target_labeled": self._child_state_dict(self.target_labeled),
            "target_unlabeled": self._child_state_dict(self.target_unlabeled),
        }

    def load_state_dict(self, state: Any) -> None:
        """Restore nested dataset histories and this worker's augmentation RNG."""
        if not isinstance(state, Mapping):
            raise TypeError("Semi-supervised worker state must be a mapping")
        if state.get("version") != _WORKER_STATE_VERSION:
            raise ValueError(
                "Unsupported semi-supervised worker state version: "
                f"{state.get('version')!r}"
            )

        python_rng = _decode_python_rng_state(state.get("python_rng"))
        numpy_rng = _decode_numpy_rng_state(state.get("numpy_rng"))
        torch_rng = _decode_torch_rng_state(state.get("torch_rng"))

        children = (
            ("source", self.source),
            ("target_labeled", self.target_labeled),
            ("target_unlabeled", self.target_unlabeled),
        )
        for field, dataset in children:
            child_state = state.get(field)
            if (dataset is None) != (child_state is None):
                raise ValueError(
                    f"Semi-supervised dataset branch {field} does not match worker state"
                )
            if dataset is not None:
                load_state_dict = getattr(dataset, "load_state_dict", None)
                if not callable(load_state_dict):
                    raise TypeError(
                        f"Nested dataset {type(dataset).__name__} must define "
                        "load_state_dict()"
                    )
                load_state_dict(child_state)

        # Nested datasets share these process-global RNGs. Restore the outer
        # worker snapshot last so child restore ordering cannot change the next
        # weak/strong augmentation draw.
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        result = {}

        # Source sample (always present, with annotations)
        source_idx = idx % len(self.source)
        result["source"] = self.source[source_idx]

        # Target labeled sample (Stage B+)
        if self.target_labeled is not None:
            tgt_idx = idx % len(self.target_labeled)
            result["target_labeled"] = self.target_labeled[tgt_idx]

        # Target unlabeled sample with weak + strong pair (Stage C+)
        if self.target_unlabeled is not None:
            tgt_idx = idx % len(self.target_unlabeled)
            raw = self._strip_unlabeled_targets(self.target_unlabeled[tgt_idx])
            weak_sample, strong_sample = self._build_target_views(raw)
            if weak_sample is not None:
                self._assert_unlabeled_sample(weak_sample, "target_weak")
                result["target_weak"] = weak_sample
            if strong_sample is not None:
                self._assert_unlabeled_sample(strong_sample, "target_strong")
                result["target_strong"] = strong_sample

        return result

    def _build_target_views(self, raw: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if self.weak_transform is None and self.strong_transform is None:
            return copy.deepcopy(raw), copy.deepcopy(raw)

        weak_split = self._split_transform(self.weak_transform) if self.weak_transform is not None else ([], [])
        strong_split = self._split_transform(self.strong_transform) if self.strong_transform is not None else ([], [])
        weak_geometry, weak_remainder = weak_split
        _strong_geometry, strong_remainder = strong_split

        geometry_sample = self._apply_steps(copy.deepcopy(raw), weak_geometry)
        weak_sample = None
        strong_sample = None
        if self.weak_transform is not None:
            weak_sample = self._apply_steps(copy.deepcopy(geometry_sample), weak_remainder)
        if self.strong_transform is not None:
            strong_sample = self._apply_steps(copy.deepcopy(geometry_sample), strong_remainder)
        return weak_sample, strong_sample

    @staticmethod
    def _transform_steps(transform: Any) -> Optional[List[Any]]:
        if transform is None:
            return []
        inner = getattr(transform, "transform", None)
        if inner is not None and hasattr(inner, "transforms"):
            return list(inner.transforms)
        if hasattr(transform, "transforms"):
            return list(transform.transforms)
        return None

    @classmethod
    def _split_transform(cls, transform: Any) -> Tuple[List[Any], List[Any]]:
        steps = cls._transform_steps(transform)
        if steps is None:
            raise TypeError(
                "VC-SUDA target weak/strong transforms must expose an ordered transforms list "
                "so geometry can be shared exactly."
            )
        geometry = []
        remainder = []
        in_remainder = False
        for step in steps:
            is_geometry = step.__class__.__name__ in _GEOMETRY_TRANSFORM_NAMES
            if is_geometry and not in_remainder:
                geometry.append(step)
            else:
                in_remainder = True
                if is_geometry:
                    raise ValueError(
                        "VC-SUDA target transforms must keep all geometry steps before non-geometry steps."
                    )
                remainder.append(step)
        return geometry, remainder

    @staticmethod
    def _apply_steps(sample: Dict[str, Any], steps: Sequence[Any]) -> Dict[str, Any]:
        for step in steps:
            sample = step(sample)
        return sample

    @staticmethod
    def _strip_unlabeled_targets(sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = copy.deepcopy(sample)
        for key in _LABEL_KEYS:
            sample.pop(key, None)
        return sample

    @staticmethod
    def _assert_unlabeled_sample(sample: Dict[str, Any], view_name: str) -> None:
        leaked = [key for key in _LABEL_KEYS if key in sample]
        if leaked:
            raise ValueError(
                f"{view_name} must be image-only for VC-SUDA unlabeled training; "
                f"found label fields {leaked}."
            )

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate that groups source/target samples separately."""
        result = {}
        result.update(SemiSupervisedDataset._collate_view(batch, "source", "source", include_annotations=True))

        if "target_labeled" in batch[0]:
            result.update(
                SemiSupervisedDataset._collate_view(
                    batch, "target_labeled", "target_labeled", include_annotations=True
                )
            )

        if "target_weak" in batch[0]:
            result.update(
                SemiSupervisedDataset._collate_view(
                    batch, "target_weak", "target_weak", include_annotations=False
                )
            )
            result.update(
                SemiSupervisedDataset._collate_view(
                    batch, "target_strong", "target_strong", include_annotations=False
                )
            )

        return result

    @staticmethod
    def _collate_view(
        batch: List[Dict[str, Any]],
        sample_key: str,
        prefix: str,
        include_annotations: bool,
    ) -> Dict[str, Any]:
        samples = [item[sample_key] for item in batch]
        result = {
            f"{prefix}_images": torch.stack([item["image"] for item in samples]),
            f"{prefix}_depths": torch.stack([item["depth"] for item in samples]),
        }

        image_ids = [item.get("image_id", 0) for item in samples]
        result[f"{prefix}_image_ids"] = torch.tensor(image_ids, dtype=torch.long)

        heights = [int(item.get("height", item["image"].shape[-2])) for item in samples]
        widths = [int(item.get("width", item["image"].shape[-1])) for item in samples]
        result[f"{prefix}_heights"] = torch.tensor(heights, dtype=torch.long)
        result[f"{prefix}_widths"] = torch.tensor(widths, dtype=torch.long)
        result[f"{prefix}_orig_sizes"] = torch.tensor(list(zip(heights, widths)), dtype=torch.long)

        content_masks = SemiSupervisedDataset._stack_optional(samples, "content_mask", dtype=torch.bool)
        if content_masks is not None:
            result[f"{prefix}_content_masks"] = content_masks
            result[f"{prefix}_padding_masks"] = ~content_masks

        noise_masks = SemiSupervisedDataset._stack_optional(samples, "noise_mask", dtype=torch.float32)
        if noise_masks is not None:
            result[f"{prefix}_noise_masks"] = noise_masks

        if include_annotations:
            result[f"{prefix}_annotations"] = [
                SemiSupervisedDataset._annotation_from_sample(sample, height, width)
                for sample, height, width in zip(samples, heights, widths)
                if "labels" in sample
            ]

        return result

    @staticmethod
    def _stack_optional(samples: List[Dict[str, Any]], key: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if key not in samples[0]:
            return None
        values = []
        for sample in samples:
            value = sample[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            values.append(value.to(dtype=dtype))
        return torch.stack(values)

    @staticmethod
    def _annotation_from_sample(sample: Dict[str, Any], height: int, width: int) -> Dict[str, Any]:
        annotation = {
            "labels": sample["labels"],
            "masks": sample["masks"],
            "boxes": sample["boxes"],
            "image_id": sample.get("image_id", 0),
            "height": height,
            "width": width,
            "orig_size": torch.tensor([height, width], dtype=torch.long),
        }
        if "content_mask" in sample:
            content_mask = sample["content_mask"].bool() if torch.is_tensor(sample["content_mask"]) else torch.as_tensor(sample["content_mask"], dtype=torch.bool)
            annotation["content_mask"] = content_mask
            annotation["padding_mask"] = ~content_mask
        if "noise_mask" in sample:
            noise_mask = sample["noise_mask"].float() if torch.is_tensor(sample["noise_mask"]) else torch.as_tensor(sample["noise_mask"], dtype=torch.float32)
            annotation["noise_mask"] = noise_mask
        return annotation
