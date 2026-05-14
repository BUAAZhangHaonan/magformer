from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from magformer.config import load_config


class _TinyEvalDataset(torch.utils.data.Dataset):
    def __init__(self, size: int = 8) -> None:
        self.size = size
        self.transform = None
        self.image_ids = list(range(100, 100 + size))
        self.coco = object()
        self.category_ids = [1]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return {
            "images": torch.zeros(3, 8, 8),
            "depths": torch.zeros(1, 8, 8),
            "image_ids": self.image_ids[index],
        }


def _runtime_cfg(eval_max_images: int, eval_batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            dataset_root="/tmp/dataset",
            val_ann="annotations/instances_val.json",
            val_split="val",
            image_size=8,
            min_scale=1.0,
            max_scale=1.0,
            depth=SimpleNamespace(
                scale=1.0,
                shift=0.0,
                clip_min=0.0,
                clip_max=1.0,
                norm="minmax",
                per_sample_norm=True,
            ),
            rgb_photo_aug=SimpleNamespace(
                brightness=0.0,
                contrast=0.0,
                saturation=0.0,
                hue=0.0,
            ),
            random_flip="none",
            depth_noise=SimpleNamespace(
                gaussian_std=0.0,
                speckle_std=0.0,
                drop_prob=0.0,
                drop_val=0.0,
            ),
        ),
        runtime=SimpleNamespace(
            eval_max_images=eval_max_images,
            eval_batch_size=eval_batch_size,
        ),
    )


def test_evaluate_build_val_loader_applies_global_eval_subset(monkeypatch) -> None:
    from tools import evaluate as evaluate_tool

    monkeypatch.setattr(
        evaluate_tool,
        "CocoRgbdDataset",
        lambda *args, **kwargs: _TinyEvalDataset(size=8),
    )
    monkeypatch.setattr(
        evaluate_tool,
        "RGBDTransform",
        lambda *args, **kwargs: ("transform", args, kwargs),
    )

    dataset, loader = evaluate_tool.build_val_loader(
        _runtime_cfg(eval_max_images=3, eval_batch_size=4),
        num_workers=0,
        batch_size=4,
    )

    assert len(dataset) == 3
    assert list(dataset.image_ids) == [100, 101, 102]
    assert loader.batch_size == 4
    assert len(loader.dataset) == 3
    assert list(loader.dataset.image_ids) == [100, 101, 102]
    assert loader.dataset.coco is dataset.coco


def test_train_build_data_loaders_subsets_val_dataset_before_distributed_sampler(
    monkeypatch,
) -> None:
    from tools import train as train_tool
    import torch.utils.data.distributed as distributed_data

    captured = {}

    class _CapturingSampler:
        def __init__(self, dataset, shuffle=False):
            captured["dataset_len"] = len(dataset)
            captured["image_ids"] = list(getattr(dataset, "image_ids", []))
            self.dataset = dataset
            self.shuffle = shuffle

        def __iter__(self):
            return iter(range(len(self.dataset)))

        def __len__(self) -> int:
            return len(self.dataset)

    monkeypatch.setattr(train_tool, "is_vc_suda_enabled", lambda config: False)
    monkeypatch.setattr(
        "magformer.data.transforms.RGBDTransform",
        lambda *args, **kwargs: ("transform", args, kwargs),
    )
    monkeypatch.setattr(distributed_data, "DistributedSampler", _CapturingSampler)

    config = _runtime_cfg(eval_max_images=3, eval_batch_size=2)
    train_dataset = _TinyEvalDataset(size=6)
    val_dataset = _TinyEvalDataset(size=8)

    _, val_loader = train_tool.build_data_loaders(
        config,
        train_dataset,
        val_dataset,
        batch_size=2,
        num_workers=0,
        is_distributed=True,
    )

    assert captured["dataset_len"] == 3
    assert captured["image_ids"] == [100, 101, 102]
    assert len(val_loader.dataset) == 3
    assert list(val_loader.dataset.image_ids) == [100, 101, 102]


def test_fast_eval_subset_configs_parse() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    cfg_200 = load_config(str(repo_root / "configs" / "eval_full_1566_fast_bbox_200.yaml"))
    cfg_300 = load_config(str(repo_root / "configs" / "eval_full_1566_fast_bbox.yaml"))

    assert cfg_200.runtime.eval_iou_types == ["bbox"]
    assert cfg_200.runtime.eval_max_images == 200
    assert cfg_200.runtime.eval_batch_size == 4
    assert cfg_300.runtime.eval_iou_types == ["bbox"]
    assert cfg_300.runtime.eval_max_images == 300
    assert cfg_300.runtime.eval_batch_size == 4
