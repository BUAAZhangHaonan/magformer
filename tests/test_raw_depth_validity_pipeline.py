from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from magformer.data.collate import collate_fn
from magformer.data.dataset import CocoRgbdDataset
from magformer.data.semi_supervised_dataset import SemiSupervisedDataset
from magformer.data.transforms import Compose, DepthNormalize, FixedSizeCrop, RandomFlip, ResizeScale, ToTensor
from magformer.models.magformer.fusion import DepthPriorExtractor, ModalityFusionModule


def _write_unlabeled_rgbd(root: Path, depth: np.ndarray) -> None:
    image_dir = root / "images" / "train"
    depth_dir = root / "depth" / "depth_npy" / "train"
    image_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    Image.fromarray(image).save(image_dir / "sample.png")
    np.save(depth_dir / "sample.npy", depth.astype(np.float32))


def test_dataset_transform_keeps_raw_valid_at_normalized_depth_boundaries(tmp_path: Path) -> None:
    raw_depth = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    _write_unlabeled_rgbd(tmp_path, raw_depth)

    dataset = CocoRgbdDataset(
        dataset_root=str(tmp_path),
        ann_file="unused.json",
        split="train",
        transform=Compose([
            DepthNormalize(scale=1.0, clip_min=0.0, clip_max=10.0, per_sample_norm=True),
            ToTensor(),
        ]),
        is_train=False,
        has_annotations=False,
    )

    sample = dataset[0]

    assert sample["depth"].shape == (1, 2, 2)
    assert sample["depth"][0, 0, 0].item() == pytest.approx(0.0)
    assert sample["depth"][0, 1, 1].item() == pytest.approx(1.0)
    assert sample["depth_valid_mask"].shape == (1, 2, 2)
    assert sample["depth_valid_mask"].dtype == torch.bool
    assert torch.equal(sample["depth_valid_mask"], torch.ones((1, 2, 2), dtype=torch.bool))


def test_depth_valid_mask_uses_nearest_geometry_for_resize_crop_and_flip() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0
    raw_valid = np.array(
        [
            [True, False, True],
            [False, True, False],
        ],
        dtype=bool,
    )
    sample = {
        "image": image,
        "depth": depth,
        "depth_valid_mask": raw_valid.copy(),
    }

    random.seed(0)
    transformed = Compose([
        RandomFlip(horizontal=True, prob=1.0),
        ResizeScale(min_scale=1.0, max_scale=1.0, target_size=6),
        FixedSizeCrop((3, 4), random_crop=False),
        ToTensor(),
    ])(sample)

    expected = np.fliplr(raw_valid).copy()
    expected = cv2.resize(expected.astype(np.uint8), (6, 4), interpolation=cv2.INTER_NEAREST).astype(bool)
    top = (4 - 3) // 2
    left = (6 - 4) // 2
    expected = expected[top : top + 3, left : left + 4]

    assert transformed["depth_valid_mask"].shape == (1, 3, 4)
    assert transformed["depth_valid_mask"].dtype == torch.bool
    assert torch.equal(transformed["depth_valid_mask"], torch.from_numpy(expected).unsqueeze(0))
    assert not transformed["depth_valid_mask"][0, 0, 1].item()
    assert not transformed["depth_valid_mask"][0, 2, 0].item()


def test_collate_batches_depth_valid_masks() -> None:
    batch = [
        {
            "image": torch.zeros((3, 2, 2), dtype=torch.float32),
            "depth": torch.zeros((1, 2, 2), dtype=torch.float32),
            "depth_valid_mask": torch.tensor([[[True, False], [True, True]]]),
            "image_id": 1,
        },
        {
            "image": torch.ones((3, 2, 2), dtype=torch.float32),
            "depth": torch.ones((1, 2, 2), dtype=torch.float32),
            "depth_valid_mask": torch.tensor([[[False, False], [True, True]]]),
            "image_id": 2,
        },
    ]

    collated = collate_fn(batch)

    assert "depth_valid_masks" in collated
    assert collated["depth_valid_masks"].shape == (2, 1, 2, 2)
    assert collated["depth_valid_masks"].dtype == torch.bool
    assert torch.equal(collated["depth_valid_masks"][0], batch[0]["depth_valid_mask"])


def test_semi_supervised_target_views_carry_transformed_depth_valid_mask() -> None:
    transform = Compose([FixedSizeCrop((2, 2), random_crop=False), ToTensor()])
    dataset = SemiSupervisedDataset.__new__(SemiSupervisedDataset)
    dataset.weak_transform = transform
    dataset.strong_transform = transform

    raw = {
        "image": np.zeros((3, 3, 3), dtype=np.uint8),
        "depth": np.ones((3, 3), dtype=np.float32),
        "depth_valid_mask": np.array(
            [
                [True, False, True],
                [True, True, False],
                [False, True, True],
            ],
            dtype=bool,
        ),
        "image_id": 7,
    }

    weak, strong = dataset._build_target_views(raw)

    assert weak is not None
    assert strong is not None
    assert "depth_valid_mask" in weak
    assert "depth_valid_mask" in strong
    assert weak["depth_valid_mask"].shape == (1, 2, 2)
    assert torch.equal(weak["depth_valid_mask"], strong["depth_valid_mask"])
    assert torch.equal(
        weak["depth_valid_mask"],
        torch.tensor([[[True, False], [True, True]]], dtype=torch.bool),
    )

    collated = SemiSupervisedDataset.collate_fn([
        {
            "source": weak,
            "target_weak": weak,
            "target_strong": strong,
        }
    ])
    assert collated["target_weak_depth_valid_masks"].shape == (1, 1, 2, 2)
    assert torch.equal(collated["target_weak_depth_valid_masks"][0], weak["depth_valid_mask"])
    assert torch.equal(collated["target_strong_depth_valid_masks"][0], strong["depth_valid_mask"])


def test_depth_prior_valid_hole_uses_raw_validity_not_normalized_depth_range() -> None:
    extractor = DepthPriorExtractor(use_rgb_edge=False)
    normalized_depth = torch.tensor([[[[0.0, 0.5], [1.0, 0.2]]]], dtype=torch.float32)
    raw_validity = torch.tensor([[[[True, True], [True, False]]]])

    priors = extractor(normalized_depth, depth_valid_mask=raw_validity)

    assert torch.equal(priors["valid"], raw_validity.float())
    assert torch.equal(priors["hole"], (~raw_validity).float())


def test_valid_hole_prior_requires_depth_valid_mask() -> None:
    fusion = ModalityFusionModule(
        image_feature_dims=[4],
        depth_feature_dims=[4],
        scale_keys=["res3"],
        fuse_scales=["res3"],
        hidden_dim=16,
        prior_enabled=True,
        prior_use_grad=False,
        prior_use_var=False,
        prior_use_valid_hole=True,
        prior_use_rgb_edge=False,
        mode="direct_add",
    )
    image_features = {"res3": torch.zeros((1, 4, 2, 2), dtype=torch.float32)}
    depth_features = {"res3": torch.zeros((1, 4, 2, 2), dtype=torch.float32)}
    depth = torch.tensor([[[[0.0, 0.4], [1.0, 0.7]]]], dtype=torch.float32)

    with pytest.raises(ValueError, match="depth_valid_masks"):
        fusion(
            image_features=image_features,
            depth_features=depth_features,
            depth_raw=depth,
        )


def test_fusion_valid_hole_marks_depth_boundaries_valid_when_raw_validity_is_true() -> None:
    fusion = ModalityFusionModule(
        image_feature_dims=[4],
        depth_feature_dims=[4],
        scale_keys=["res3"],
        fuse_scales=["res3"],
        hidden_dim=16,
        prior_enabled=True,
        prior_use_grad=False,
        prior_use_var=False,
        prior_use_valid_hole=True,
        prior_use_rgb_edge=False,
        mode="direct_add",
    )
    depth = torch.tensor([[[[0.0, 0.4], [1.0, 0.7]]]], dtype=torch.float32)
    raw_validity = torch.tensor([[[[True, True], [True, False]]]])
    priors = fusion._prepare_priors_ms(
        depth,
        rgb_image=None,
        target_sizes={"res3": (2, 2)},
        depth_valid_mask=raw_validity,
    )

    assert torch.equal(priors["res3"]["valid"], raw_validity.float())
    assert torch.equal(priors["res3"]["hole"], (~raw_validity).float())
