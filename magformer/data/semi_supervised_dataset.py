# -*- coding: utf-8 -*-
"""
Semi-Supervised Dataset for VC-SUDA

Combines labeled source data with unlabeled target data,
providing weak/strong augmentation pairs for teacher-student training.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence, Tuple
import copy
import json
import math

import torch
from torch.utils.data import Dataset

from .dataset import CocoRgbdDataset


_LABEL_KEYS = ("labels", "masks", "boxes", "annotations")
_SAMPLING_FORBIDDEN_KEYS = {"gt_count", "gt_density_bucket", "annotations"}
_GEOMETRY_TRANSFORM_NAMES = {
    "InitContentMask",
    "RandomFlip",
    "ResizeScale",
    "FixedSizeCrop",
}


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
        source_root: Optional[str],
        source_ann: Optional[str],
        source_split: str = "train",
        source_datasets: Optional[Sequence[Any]] = None,
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
        target_unlabeled_sampling_stats: Optional[str] = None,
        weak_transform: Optional[Any] = None,
        strong_transform: Optional[Any] = None,
        # Stage control
        stage: str = "A",
    ):
        super().__init__()

        self.stage = stage

        # Source dataset(s) are always present, labeled, and use source augmentation.
        self._build_source_datasets(
            source_root=source_root,
            source_ann=source_ann,
            source_split=source_split,
            source_datasets=source_datasets,
            source_transform=source_transform,
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
            self.target_unlabeled_index_sequence = self._build_target_unlabeled_index_sequence(
                target_unlabeled_sampling_stats
            )
        else:
            self.target_unlabeled_index_sequence = []

        print(f"[SemiSupervisedDataset] Stage={stage}, "
              f"source={self._source_summary()}, "
              f"target_labeled={len(self.target_labeled) if self.target_labeled else 0}, "
              f"target_unlabeled={len(self.target_unlabeled) if self.target_unlabeled else 0}, "
              f"target_unlabeled_sequence={len(self.target_unlabeled_index_sequence)}")

    @staticmethod
    def _cfg_value(config: Any, key: str, default: Any = None) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _build_source_datasets(
        self,
        source_root: Optional[str],
        source_ann: Optional[str],
        source_split: str,
        source_datasets: Optional[Sequence[Any]],
        source_transform: Optional[Any],
    ) -> None:
        self.source_dataset_names: List[str] = []
        self.source_dataset_weights: List[int] = []
        self.source_datasets: List[CocoRgbdDataset] = []
        self.source_index_sequence: List[int] = []
        self.source_occurrence_offsets: List[int] = []
        self._multi_source_enabled = source_datasets is not None

        if source_datasets is None:
            if not source_root:
                raise ValueError("vc_suda.source_root is required for legacy single source")
            if not source_ann:
                raise ValueError("vc_suda.source_ann is required for legacy single source")
            self.source = CocoRgbdDataset(
                dataset_root=source_root,
                ann_file=source_ann,
                split=source_split,
                transform=source_transform,
                is_train=True,
                has_annotations=True,
            )
            if len(self.source) == 0:
                raise ValueError(f"empty source dataset: {source_ann}")
            self.source_datasets = [self.source]
            self.source_dataset_names = ["source"]
            self.source_dataset_weights = [1]
            self.source_index_sequence = [0]
            self.source_occurrence_offsets = [0]
            self._source_length = len(self.source)
            return

        if len(source_datasets) == 0:
            raise ValueError("source_datasets must not be empty when set")

        seen_names = set()
        for source_index, source_cfg in enumerate(source_datasets):
            name = self._cfg_value(source_cfg, "name")
            root = self._cfg_value(source_cfg, "root")
            ann = self._cfg_value(source_cfg, "ann")
            split = self._cfg_value(source_cfg, "split", "train")
            weight = self._cfg_value(source_cfg, "weight", 1)

            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"vc_suda.source_datasets[{source_index}].name is required")
            name = name.strip()
            if name in seen_names:
                raise ValueError(f"duplicate source_datasets name: {name}")
            seen_names.add(name)
            if not isinstance(root, str) or not root.strip():
                raise ValueError(f"vc_suda.source_datasets[{source_index}].root is required")
            if not isinstance(ann, str) or not ann.strip():
                raise ValueError(f"vc_suda.source_datasets[{source_index}].ann is required")
            if not isinstance(split, str) or not split.strip():
                raise ValueError(f"vc_suda.source_datasets[{source_index}].split is required")
            if isinstance(weight, bool):
                raise ValueError(f"vc_suda.source_datasets[{source_index}].weight must be >= 1")
            try:
                weight = int(weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"vc_suda.source_datasets[{source_index}].weight must be >= 1"
                ) from exc
            if weight < 1:
                raise ValueError(f"vc_suda.source_datasets[{source_index}].weight must be >= 1")

            dataset = CocoRgbdDataset(
                dataset_root=root,
                ann_file=ann,
                split=split,
                transform=source_transform,
                is_train=True,
                has_annotations=True,
            )
            if len(dataset) == 0:
                raise ValueError(f"empty source dataset: {name}")
            self.source_datasets.append(dataset)
            self.source_dataset_names.append(name)
            self.source_dataset_weights.append(weight)

        self.source = self.source_datasets[0]
        counts = [0 for _ in self.source_datasets]
        for source_index, weight in enumerate(self.source_dataset_weights):
            for _ in range(weight):
                self.source_index_sequence.append(source_index)
                self.source_occurrence_offsets.append(counts[source_index])
                counts[source_index] += 1

        cycle_count = max(
            math.ceil(len(dataset) / weight)
            for dataset, weight in zip(self.source_datasets, self.source_dataset_weights)
        )
        self._source_length = cycle_count * len(self.source_index_sequence)

    def _source_summary(self) -> str:
        if not self._multi_source_enabled:
            return str(len(self.source))
        return ",".join(
            f"{name}:{len(dataset)}x{weight}"
            for name, dataset, weight in zip(
                self.source_dataset_names,
                self.source_datasets,
                self.source_dataset_weights,
            )
        )

    def set_source_transform(self, transform: Any) -> None:
        """Apply the same source transform to every source dataset."""
        for dataset in self.source_datasets:
            dataset.transform = transform

    def _get_source_sample(self, idx: int) -> Dict[str, Any]:
        sequence_pos = idx % len(self.source_index_sequence)
        source_dataset_index = self.source_index_sequence[sequence_pos]
        dataset = self.source_datasets[source_dataset_index]
        cycle_index = idx // len(self.source_index_sequence)
        local_occurrence = (
            cycle_index * self.source_dataset_weights[source_dataset_index]
            + self.source_occurrence_offsets[sequence_pos]
        )
        sample = dataset[local_occurrence % len(dataset)]
        if self._multi_source_enabled:
            sample["source_dataset_name"] = self.source_dataset_names[source_dataset_index]
            sample["source_dataset_index"] = source_dataset_index
        return sample

    def __len__(self) -> int:
        # Length is determined by the larger of source coverage and unlabeled target coverage.
        lengths = [self._source_length]
        if self.target_unlabeled is not None:
            lengths.append(len(self.target_unlabeled_index_sequence))
        return max(lengths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        result = {}

        # Source sample (always present, with annotations)
        result["source"] = self._get_source_sample(idx)

        # Target labeled sample (Stage B+)
        if self.target_labeled is not None:
            tgt_idx = idx % len(self.target_labeled)
            result["target_labeled"] = self.target_labeled[tgt_idx]

        # Target unlabeled sample with weak + strong pair (Stage C+)
        if self.target_unlabeled is not None:
            sequence_idx = idx % len(self.target_unlabeled_index_sequence)
            tgt_idx = self.target_unlabeled_index_sequence[sequence_idx]
            raw = self._strip_unlabeled_targets(self.target_unlabeled[tgt_idx])
            weak_sample, strong_sample = self._build_target_views(raw)
            if weak_sample is not None:
                self._assert_unlabeled_sample(weak_sample, "target_weak")
                result["target_weak"] = weak_sample
            if strong_sample is not None:
                self._assert_unlabeled_sample(strong_sample, "target_strong")
                result["target_strong"] = strong_sample

        return result

    def _build_target_unlabeled_index_sequence(self, stats_path: Optional[str]) -> List[int]:
        if self.target_unlabeled is None:
            return []
        if not stats_path:
            return list(range(len(self.target_unlabeled)))

        payload_path = Path(stats_path)
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        self._assert_sampling_stats_prediction_only(payload, str(payload_path))
        rows = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"target_unlabeled sampling stats must contain an images list: {payload_path}")

        image_id_to_index = {int(image_id): idx for idx, image_id in enumerate(self.target_unlabeled.image_ids)}
        sequence: List[int] = []
        for row_idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"target_unlabeled sampling stats images[{row_idx}] must be an object")
            if "image_id" not in row:
                raise ValueError(f"target_unlabeled sampling stats images[{row_idx}] missing image_id")
            image_id = int(row["image_id"])
            if image_id not in image_id_to_index:
                raise ValueError(
                    f"target_unlabeled sampling stats image_id {image_id} is not present in target dataset"
                )
            repeat = int(row.get("repeat", 1))
            if repeat < 1:
                raise ValueError(f"target_unlabeled sampling repeat must be >= 1 for image_id {image_id}")
            sequence.extend([image_id_to_index[image_id]] * repeat)

        if not sequence:
            raise ValueError("target_unlabeled sampling stats produced an empty index sequence")
        return sequence

    @classmethod
    def _assert_sampling_stats_prediction_only(cls, value: Any, context: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in _SAMPLING_FORBIDDEN_KEYS:
                    raise ValueError(f"{context} contains forbidden target sampling field: {key}")
                cls._assert_sampling_stats_prediction_only(child, f"{context}.{key}")
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                cls._assert_sampling_stats_prediction_only(child, f"{context}[{idx}]")

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
