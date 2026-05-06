# -*- coding: utf-8 -*-
"""
Semi-Supervised Dataset for VC-SUDA

Combines labeled source data with unlabeled target data,
providing weak/strong augmentation pairs for teacher-student training.
"""

from typing import Dict, List, Optional, Any
import copy

import torch
from torch.utils.data import Dataset, ConcatDataset

from .dataset import CocoRgbdDataset
from .transforms import RGBDTransform, get_weak_augmentation


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
            # Base dataset without transform — we apply transforms manually
            # Use has_annotations=True so COCO parsing determines which images
            # to load (the 228 target_unlabeled images), but is_train=False so
            # annotation labels are not loaded during training.
            self._target_unlabeled_base = CocoRgbdDataset(
                dataset_root=target_unlabeled_root,
                ann_file=target_unlabeled_ann,
                split=target_unlabeled_split,
                transform=None,  # No transform — we apply weak/strong manually
                is_train=False,  # Don't load annotation labels
                has_annotations=True,  # Use annotation file for image listing
            )
            self.target_unlabeled = self._target_unlabeled_base

        self.stage = stage
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
            raw = self.target_unlabeled[tgt_idx]

            # Apply weak augmentation (for teacher)
            if self.weak_transform is not None:
                raw_copy = copy.deepcopy(raw)
                result["target_weak"] = self.weak_transform(raw_copy)

            # Apply strong augmentation (for student)
            if self.strong_transform is not None:
                raw_copy = copy.deepcopy(raw)
                result["target_strong"] = self.strong_transform(raw_copy)

        return result

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate that groups source/target samples separately."""
        result = {}

        # Source samples
        source_images = torch.stack([item["source"]["image"] for item in batch])
        source_depths = torch.stack([item["source"]["depth"] for item in batch])
        result["source_images"] = source_images
        result["source_depths"] = source_depths
        result["source_annotations"] = [
            {
                "labels": item["source"]["labels"],
                "masks": item["source"]["masks"],
                "boxes": item["source"]["boxes"],
            }
            for item in batch
            if "labels" in item["source"]
        ]

        # Target labeled samples (Stage B+)
        if "target_labeled" in batch[0]:
            result["target_labeled_images"] = torch.stack(
                [item["target_labeled"]["image"] for item in batch]
            )
            result["target_labeled_depths"] = torch.stack(
                [item["target_labeled"]["depth"] for item in batch]
            )
            result["target_labeled_annotations"] = [
                {
                    "labels": item["target_labeled"]["labels"],
                    "masks": item["target_labeled"]["masks"],
                    "boxes": item["target_labeled"]["boxes"],
                }
                for item in batch
                if "labels" in item["target_labeled"]
            ]

        # Target unlabeled samples (Stage C+)
        if "target_weak" in batch[0]:
            result["target_weak_images"] = torch.stack(
                [item["target_weak"]["image"] for item in batch]
            )
            result["target_weak_depths"] = torch.stack(
                [item["target_weak"]["depth"] for item in batch]
            )

            result["target_strong_images"] = torch.stack(
                [item["target_strong"]["image"] for item in batch]
            )
            result["target_strong_depths"] = torch.stack(
                [item["target_strong"]["depth"] for item in batch]
            )

        return result
