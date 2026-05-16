from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from magformer.config.vc_suda_schema import VCSUDAConfig
from magformer.data.dataset import CocoRgbdDataset as RealCocoRgbdDataset
from magformer.data.semi_supervised_dataset import SemiSupervisedDataset
from magformer.data.transforms import Compose, ToTensor


def _numpy_sample(image_id: int, with_labels: bool = True):
    image = np.full((4, 4, 3), image_id % 255, dtype=np.uint8)
    depth = np.full((4, 4), float(image_id), dtype=np.float32)
    sample = {
        "image": image,
        "depth": depth,
        "image_id": image_id,
        "height": 4,
        "width": 4,
    }
    if with_labels:
        sample.update(
            {
                "labels": np.array([1], dtype=np.int64),
                "masks": np.ones((4, 4, 1), dtype=bool),
                "boxes": np.array([[0, 0, 3, 3]], dtype=np.float32),
            }
        )
    return sample


class _FakeCocoRgbdDataset:
    instances = []
    samples_by_ann = {
        "annotations/source_a.json": [_numpy_sample(100), _numpy_sample(101), _numpy_sample(102)],
        "annotations/source_b.json": [_numpy_sample(200), _numpy_sample(201)],
        "annotations/legacy_source.json": [_numpy_sample(300), _numpy_sample(301)],
        "annotations/target_labeled.json": [_numpy_sample(400)],
        "annotations/target_unlabeled.json": [_numpy_sample(500), _numpy_sample(501), _numpy_sample(502)],
        "annotations/empty_source.json": [],
    }

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.samples = list(self.samples_by_ann[kwargs["ann_file"]])
        self.image_ids = [sample["image_id"] for sample in self.samples]
        _FakeCocoRgbdDataset.instances.append(self)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.samples[idx])
        transform = self.kwargs.get("transform")
        if transform is not None:
            sample = transform(sample)
        return sample


@pytest.fixture(autouse=True)
def _fake_coco(monkeypatch):
    import magformer.data.semi_supervised_dataset as semi_module

    _FakeCocoRgbdDataset.instances = []
    monkeypatch.setattr(semi_module, "CocoRgbdDataset", _FakeCocoRgbdDataset)


def test_legacy_single_source_config_keeps_existing_source_sampling():
    dataset = SemiSupervisedDataset(
        source_root="/tmp/legacy_source",
        source_ann="annotations/legacy_source.json",
        source_split="train",
        target_unlabeled_root="/tmp/target",
        target_unlabeled_ann="annotations/target_unlabeled.json",
        stage="C",
    )

    assert len(dataset.source) == 2
    assert len(dataset) == 3
    assert [dataset[idx]["source"]["image_id"] for idx in range(4)] == [300, 301, 300, 301]
    assert "source_dataset_name" not in dataset[0]["source"]


def test_multi_source_builds_targets_and_uses_weighted_round_robin_order():
    dataset = SemiSupervisedDataset(
        source_root="/tmp/ignored_old_root",
        source_ann="annotations/legacy_source.json",
        source_datasets=[
            {"name": "original", "root": "/tmp/source_a", "ann": "annotations/source_a.json", "weight": 1},
            {"name": "pseudo_real", "root": "/tmp/source_b", "ann": "annotations/source_b.json", "weight": 2},
        ],
        target_labeled_root="/tmp/target",
        target_labeled_ann="annotations/target_labeled.json",
        target_unlabeled_root="/tmp/target",
        target_unlabeled_ann="annotations/target_unlabeled.json",
        stage="C",
    )

    source_loads = [instance.kwargs for instance in _FakeCocoRgbdDataset.instances[:2]]
    assert [kwargs["dataset_root"] for kwargs in source_loads] == ["/tmp/source_a", "/tmp/source_b"]
    assert [kwargs["ann_file"] for kwargs in source_loads] == [
        "annotations/source_a.json",
        "annotations/source_b.json",
    ]
    assert dataset.target_labeled is not None
    assert dataset.target_unlabeled is not None

    # Length is max(ceil(source_len / source_weight)) * sum(weights), then
    # compared with target_unlabeled sequence length. This covers every source.
    assert len(dataset) == 9

    samples = [dataset[idx] for idx in range(7)]
    assert [sample["source"]["source_dataset_name"] for sample in samples] == [
        "original",
        "pseudo_real",
        "pseudo_real",
        "original",
        "pseudo_real",
        "pseudo_real",
        "original",
    ]
    assert [sample["source"]["image_id"] for sample in samples] == [100, 200, 201, 101, 200, 201, 102]
    assert [sample["target_labeled"]["image_id"] for sample in samples[:3]] == [400, 400, 400]
    assert [sample["target_weak"]["image_id"] for sample in samples[:4]] == [500, 501, 502, 500]


@pytest.mark.parametrize(
    ("source_datasets", "match"),
    [
        ([], "source_datasets must not be empty"),
        ([{"name": "bad", "root": "/tmp/source_a", "ann": "annotations/source_a.json", "weight": 0}], "weight"),
        (
            [
                {"name": "dup", "root": "/tmp/source_a", "ann": "annotations/source_a.json"},
                {"name": "dup", "root": "/tmp/source_b", "ann": "annotations/source_b.json"},
            ],
            "duplicate source_datasets name",
        ),
        ([{"name": "empty", "root": "/tmp/source_a", "ann": "annotations/empty_source.json"}], "empty source"),
    ],
)
def test_multi_source_invalid_configs_fail_fast(source_datasets, match):
    with pytest.raises(ValueError, match=match):
        SemiSupervisedDataset(
            source_root="/tmp/legacy_source",
            source_ann="annotations/legacy_source.json",
            source_datasets=source_datasets,
            stage="A",
        )


def test_vc_suda_schema_validates_source_datasets():
    cfg = VCSUDAConfig(
        enabled=True,
        source_datasets=[
            {"name": "original", "root": "magformer_datasets/20260318_1K_1566", "ann": "annotations/instances_train.json"},
            {
                "name": "pseudo",
                "root": "magformer_datasets/pseudo_real_512",
                "ann": "annotations/instances_source.json",
                "split": "train",
                "weight": 2,
            },
        ],
    )

    assert cfg.source_datasets is not None
    assert cfg.source_datasets[0].split == "train"
    assert cfg.source_datasets[0].weight == 1
    assert cfg.source_datasets[1].weight == 2

    with pytest.raises(ValueError, match="source_datasets must not be empty"):
        VCSUDAConfig(enabled=True, source_datasets=[])
    with pytest.raises(ValueError, match="weight"):
        VCSUDAConfig(
            enabled=True,
            source_datasets=[{"name": "bad", "root": "root", "ann": "ann", "weight": 0}],
        )
    with pytest.raises(ValueError, match="duplicate source_datasets name"):
        VCSUDAConfig(
            enabled=True,
            source_datasets=[
                {"name": "dup", "root": "root_a", "ann": "ann_a"},
                {"name": "dup", "root": "root_b", "ann": "ann_b"},
            ],
        )
    with pytest.raises(ValueError, match="root"):
        VCSUDAConfig(enabled=True, source_datasets=[{"name": "missing", "ann": "ann"}])


def test_real_multi_source_smoke_can_fetch_and_collate_items(monkeypatch):
    import magformer.data.semi_supervised_dataset as semi_module

    monkeypatch.setattr(semi_module, "CocoRgbdDataset", RealCocoRgbdDataset)
    transform = Compose([ToTensor()])
    dataset = SemiSupervisedDataset(
        source_root="ignored",
        source_ann="ignored.json",
        source_datasets=[
            {
                "name": "original",
                "root": "magformer_datasets/20260318_1K_1566",
                "ann": "annotations/instances_train.json",
                "weight": 1,
            },
            {
                "name": "pseudo_real",
                "root": "magformer_datasets/pseudo_real_512",
                "ann": "annotations/instances_source.json",
                "weight": 1,
            },
        ],
        source_transform=transform,
        target_labeled_root="magformer_datasets/pseudo_real_512",
        target_labeled_ann="annotations/instances_target_labeled.json",
        target_labeled_transform=transform,
        target_unlabeled_root="magformer_datasets/pseudo_real_512",
        target_unlabeled_ann="annotations/instances_target_unlabeled.json",
        weak_transform=transform,
        strong_transform=transform,
        stage="C",
    )

    samples = [dataset[idx] for idx in range(4)]
    assert [sample["source"]["source_dataset_name"] for sample in samples] == [
        "original",
        "pseudo_real",
        "original",
        "pseudo_real",
    ]
    for sample in samples:
        assert torch.is_tensor(sample["source"]["image"])
        assert torch.is_tensor(sample["source"]["labels"])
        assert torch.is_tensor(sample["target_labeled"]["image"])
        assert torch.is_tensor(sample["target_weak"]["image"])
        assert "labels" not in sample["target_weak"]
