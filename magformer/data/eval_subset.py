from __future__ import annotations

from typing import Any, Iterable

from torch.utils.data import Dataset


class EvalSubsetDataset(Dataset):
    """A thin dataset view that preserves dataset-level metadata for eval."""

    def __init__(self, dataset: Dataset, indices: Iterable[int]) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.image_ids = self._subset_image_ids(dataset, self.indices)

    @staticmethod
    def _subset_image_ids(dataset: Dataset, indices: list[int]) -> list[int]:
        image_ids = getattr(dataset, "image_ids", None)
        if image_ids is None:
            return indices
        return [int(image_ids[index]) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def build_global_eval_subset(dataset: Dataset, max_images: int | None) -> Dataset:
    """Return a dataset-level eval subset for the first ``max_images`` entries."""
    if max_images is None:
        return dataset

    limit = int(max_images)
    if limit < 1:
        raise ValueError("eval_max_images must be >= 1 when provided")
    if limit >= len(dataset):
        return dataset

    return EvalSubsetDataset(dataset, range(limit))
