import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from magformer.config import load_config
from magformer.config.loader import load_yaml_file
from magformer.data.dataset import CocoRgbdDataset
from magformer.data.semi_supervised_dataset import SemiSupervisedDataset
from magformer.data.transforms import Compose, FixedSizeCrop, InitContentMask, RGBDTransform, ToTensor


VC_SUDA_CONFIG = "configs/vc_suda_stage_a_40ep_512.yaml"
VC_SUDA_STAGE_B_TEACHER8499_CONFIG = "configs/vc_suda_stage_b_1024_teacher8499.yaml"


def _stage_b_eval_depth_transform(cfg):
    return RGBDTransform(
        image_size=cfg.data.image_size,
        min_scale=cfg.data.min_scale,
        max_scale=cfg.data.max_scale,
        random_flip="none",
        rgb_brightness=0.0,
        rgb_contrast=0.0,
        rgb_saturation=0.0,
        rgb_hue=0.0,
        depth_scale=cfg.data.depth.scale,
        depth_shift=cfg.data.depth.shift,
        depth_clip_min=cfg.data.depth.clip_min,
        depth_clip_max=cfg.data.depth.clip_max,
        depth_norm=cfg.data.depth.norm,
        depth_per_sample_norm=cfg.data.depth.per_sample_norm,
        is_train=False,
    )


def test_stage_b_teacher8499_config_uses_teacher_architecture_and_runtime_contract():
    cfg = load_config(VC_SUDA_STAGE_B_TEACHER8499_CONFIG)

    assert cfg.vc_suda.enabled is True
    assert cfg.vc_suda.stage == "B"
    assert cfg.data.dataset_root == "magformer_datasets/pseudo_real_512"
    assert cfg.data.image_size == 1024
    assert cfg.data.train_ann == "annotations/instances_source.json"
    assert cfg.data.val_ann == "annotations/instances_val.json"
    assert cfg.vc_suda.source_ann == "annotations/instances_source.json"
    assert cfg.vc_suda.target_labeled_ann == "annotations/instances_target_labeled.json"
    assert cfg.vc_suda.target_unlabeled_ann == "annotations/instances_target_unlabeled.json"
    assert cfg.model.finetune_weights == "output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth"
    assert list(cfg.runtime.gpus) == [4, 5, 6, 7]
    assert cfg.runtime.early_stop == {"enabled": False}
    assert "stage_b_1024_teacher8499" in cfg.runtime.output_dir
    assert cfg.runtime.output_dir != "output/vc_suda/stage_a"
    assert cfg.runtime.output_dir != "/home/hdd3/zhanghaonan/magformer/output/experiments/20260510_1k_finetune_full_1024_v13"


def test_vc_suda_target_labeled_weight_defaults_and_validates():
    cfg = load_config(VC_SUDA_CONFIG)
    assert cfg.vc_suda.target_labeled_weight == pytest.approx(1.0)

    cfg = load_config(
        VC_SUDA_CONFIG,
        overrides={"vc_suda": {"target_labeled_weight": 0.5}},
    )
    assert cfg.vc_suda.target_labeled_weight == pytest.approx(0.5)

    cfg = load_config(
        VC_SUDA_CONFIG,
        overrides={"vc_suda": {"target_labeled_weight": 0.25}},
    )
    assert cfg.vc_suda.target_labeled_weight == pytest.approx(0.25)

    with pytest.raises(ValueError, match="target_labeled_weight"):
        load_config(
            VC_SUDA_CONFIG,
            overrides={"vc_suda": {"target_labeled_weight": -0.1}},
        )


def test_stage_b_teacher8499_dataset_manifests_have_expected_split_sizes():
    cfg = load_config(VC_SUDA_STAGE_B_TEACHER8499_CONFIG)
    root = Path(cfg.data.dataset_root)

    def image_count(ann_file):
        return len(load_yaml_file(root / ann_file).get("images", []))

    assert image_count(cfg.vc_suda.source_ann) == 1008
    assert image_count(cfg.vc_suda.target_labeled_ann) == 25
    assert image_count(cfg.data.val_ann) == 28


def test_stage_b_teacher8499_depth_transform_preserves_pseudo_real_variation():
    cfg = load_config(VC_SUDA_STAGE_B_TEACHER8499_CONFIG)
    transform = _stage_b_eval_depth_transform(cfg)
    datasets = {
        "source": CocoRgbdDataset(
            cfg.data.dataset_root,
            cfg.vc_suda.source_ann,
            split=cfg.data.train_split,
            transform=transform,
            is_train=False,
            has_annotations=True,
        ),
        "target_labeled": CocoRgbdDataset(
            cfg.data.dataset_root,
            cfg.vc_suda.target_labeled_ann,
            split=cfg.data.train_split,
            transform=transform,
            is_train=False,
            has_annotations=True,
        ),
        "val": CocoRgbdDataset(
            cfg.data.dataset_root,
            cfg.data.val_ann,
            split=cfg.data.val_split,
            transform=transform,
            is_train=False,
            has_annotations=True,
        ),
    }

    assert len(datasets["source"]) == 1008
    assert len(datasets["target_labeled"]) == 25
    assert len(datasets["val"]) == 28

    for split_name, dataset in datasets.items():
        for sample_idx in range(2):
            depth = dataset[sample_idx]["depth"]
            assert depth.std().item() > 1e-4, (split_name, sample_idx)
            if split_name in {"target_labeled", "val"}:
                assert torch.unique(depth).numel() > 1, (split_name, sample_idx)
                assert depth.min().item() >= 0.0, (split_name, sample_idx)
                assert depth.max().item() <= 1.0, (split_name, sample_idx)

    assert cfg.data.depth.clip_min == pytest.approx(0.0)
    assert cfg.data.depth.clip_max == pytest.approx(2.095623016357422)


def test_stage_b_teacher8499_builds_source_and_target_labeled_only_dataset():
    cfg = load_config(VC_SUDA_STAGE_B_TEACHER8499_CONFIG)

    dataset = SemiSupervisedDataset(
        source_root=cfg.data.dataset_root,
        source_ann=cfg.vc_suda.source_ann,
        source_split=cfg.data.train_split,
        target_labeled_root=cfg.data.dataset_root,
        target_labeled_ann=cfg.vc_suda.target_labeled_ann,
        target_labeled_split=cfg.data.train_split,
        target_unlabeled_root=cfg.data.dataset_root,
        target_unlabeled_ann=cfg.vc_suda.target_unlabeled_ann,
        target_unlabeled_split=cfg.data.train_split,
        stage=cfg.vc_suda.stage,
    )

    assert len(dataset.source) == 1008
    assert dataset.target_labeled is not None
    assert len(dataset.target_labeled) == 25
    assert dataset.target_unlabeled is None


def test_stage_a_config_uses_existing_source_annotation():
    cfg = load_config(VC_SUDA_CONFIG)

    assert cfg.data.train_ann == "annotations/instances_source.json"
    assert cfg.vc_suda.enabled is True
    assert cfg.vc_suda.source_ann == "annotations/instances_source.json"


def test_stage_a_config_pins_runtime_gpus_without_stale_markers():
    cfg = load_config(VC_SUDA_CONFIG)
    raw_runtime = load_yaml_file(VC_SUDA_CONFIG)["runtime"]

    runtime_keys = set(raw_runtime)
    stale_gpu_markers = {
        key for key in runtime_keys
        if key != "gpus" and "gpu" in key.lower()
    }
    config_text = Path(VC_SUDA_CONFIG).read_text(encoding="utf-8").lower()

    assert list(cfg.runtime.gpus) == [4, 5, 6, 7]
    assert stale_gpu_markers == set()
    assert runtime_keys <= set(type(cfg.runtime).model_fields)
    assert "gpus_old" not in config_text
    assert "old_gpus" not in config_text
    assert "gpu_old" not in config_text


@pytest.mark.parametrize(
    ("stage", "target_labeled_ann", "target_unlabeled_ann", "message"),
    [
        ("B", None, None, "target_labeled_ann"),
        ("C", "annotations/instances_target_labeled.json", None, "target_unlabeled_ann"),
        ("D", "annotations/instances_target_labeled.json", None, "target_unlabeled_ann"),
        ("E", "annotations/instances_target_labeled.json", None, "target_unlabeled_ann"),
    ],
)
def test_vc_suda_stage_requirements_fail_fast(stage, target_labeled_ann, target_unlabeled_ann, message):
    with pytest.raises(ValueError, match=message):
        load_config(
            VC_SUDA_CONFIG,
            overrides={
                "vc_suda": {
                    "stage": stage,
                    "source_ann": "annotations/instances_source.json",
                    "target_labeled_ann": target_labeled_ann,
                    "target_unlabeled_ann": target_unlabeled_ann,
                }
            },
        )


def test_vc_suda_stage_enum_rejects_typos():
    with pytest.raises(ValueError, match="stage"):
        load_config(VC_SUDA_CONFIG, overrides={"vc_suda": {"stage": "stage_c"}})


def test_unknown_vc_suda_and_runtime_keys_are_rejected():
    with pytest.raises(ValueError, match="stgae"):
        load_config(VC_SUDA_CONFIG, overrides={"vc_suda": {"stgae": "C"}})

    with pytest.raises(ValueError, match="unexpected_runtime_key"):
        load_config(VC_SUDA_CONFIG, overrides={"runtime": {"unexpected_runtime_key": True}})


def test_target_unlabeled_cannot_reuse_eval_annotations():
    with pytest.raises(ValueError, match="target_unlabeled_ann.*val_ann"):
        load_config(
            VC_SUDA_CONFIG,
            overrides={
                "vc_suda": {
                    "stage": "C",
                    "source_ann": "annotations/instances_source.json",
                    "target_labeled_ann": "annotations/instances_target_labeled.json",
                    "target_unlabeled_ann": "annotations/instances_val.json",
                }
            },
        )


def _tensor_sample(image_id):
    image = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    depth = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    content_mask = torch.tensor([[True, False], [True, True]])
    noise_mask = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    return {
        "image": image,
        "depth": depth,
        "image_id": image_id,
        "height": 5,
        "width": 6,
        "content_mask": content_mask,
        "noise_mask": noise_mask,
        "labels": torch.tensor([1], dtype=torch.long),
        "masks": torch.ones((1, 2, 2), dtype=torch.bool),
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
    }


def test_semi_supervised_collate_preserves_metadata_and_masks():
    batch = [
        {
            "source": _tensor_sample(11),
            "target_labeled": _tensor_sample(22),
            "target_weak": {
                **_tensor_sample(33),
                "labels": torch.tensor([], dtype=torch.long),
                "masks": torch.zeros((0, 2, 2), dtype=torch.bool),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
            },
            "target_strong": {
                **_tensor_sample(44),
                "labels": torch.tensor([], dtype=torch.long),
                "masks": torch.zeros((0, 2, 2), dtype=torch.bool),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
            },
        }
    ]

    collated = SemiSupervisedDataset.collate_fn(batch)

    assert collated["source_image_ids"].tolist() == [11]
    assert collated["source_heights"].tolist() == [5]
    assert collated["source_widths"].tolist() == [6]
    assert torch.equal(collated["source_content_masks"], batch[0]["source"]["content_mask"].unsqueeze(0))
    assert torch.equal(collated["source_padding_masks"], ~batch[0]["source"]["content_mask"].unsqueeze(0))
    assert torch.equal(collated["source_noise_masks"], batch[0]["source"]["noise_mask"].unsqueeze(0))
    assert collated["source_annotations"][0]["image_id"] == 11
    assert collated["source_annotations"][0]["height"] == 5
    assert collated["source_annotations"][0]["width"] == 6

    assert collated["target_weak_image_ids"].tolist() == [33]
    assert torch.equal(collated["target_weak_padding_masks"], ~batch[0]["target_weak"]["content_mask"].unsqueeze(0))
    assert torch.equal(collated["target_weak_noise_masks"], batch[0]["target_weak"]["noise_mask"].unsqueeze(0))
    assert collated["target_strong_image_ids"].tolist() == [44]
    assert torch.equal(collated["target_strong_padding_masks"], ~batch[0]["target_strong"]["content_mask"].unsqueeze(0))
    assert torch.equal(collated["target_strong_noise_masks"], batch[0]["target_strong"]["noise_mask"].unsqueeze(0))


class _FakeCocoRgbdDataset:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        ann_file = kwargs.get("ann_file", "")
        self.sample = kwargs.pop("sample", None) or _numpy_sample(
            with_labels=kwargs.get("is_train", False)
            or "target_unlabeled" in ann_file
        )
        _FakeCocoRgbdDataset.instances.append(self)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        del idx
        return copy.deepcopy(self.sample)


def _numpy_sample(with_labels):
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    sample = {
        "image": image,
        "depth": depth,
        "image_id": 9,
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


def _build_fake_stage_c_dataset(monkeypatch, weak_transform, strong_transform):
    import magformer.data.semi_supervised_dataset as semi_module

    _FakeCocoRgbdDataset.instances = []
    monkeypatch.setattr(semi_module, "CocoRgbdDataset", _FakeCocoRgbdDataset)
    return SemiSupervisedDataset(
        source_root="/tmp/source",
        source_ann="annotations/instances_source.json",
        source_transform=ToTensor(),
        target_labeled_root="/tmp/target",
        target_labeled_ann="annotations/instances_target_labeled.json",
        target_labeled_transform=ToTensor(),
        target_unlabeled_root="/tmp/target",
        target_unlabeled_ann="annotations/instances_target_unlabeled.json",
        weak_transform=weak_transform,
        strong_transform=strong_transform,
        stage="C",
    )


def test_target_unlabeled_samples_never_expose_ground_truth_labels(monkeypatch):
    transform = Compose([ToTensor()])
    dataset = _build_fake_stage_c_dataset(monkeypatch, transform, transform)

    sample = dataset[0]

    assert _FakeCocoRgbdDataset.instances[-1].kwargs["is_train"] is False
    assert _FakeCocoRgbdDataset.instances[-1].kwargs["has_annotations"] is True
    for view_name in ("target_weak", "target_strong"):
        assert "labels" not in sample[view_name]
        assert "masks" not in sample[view_name]
        assert "boxes" not in sample[view_name]


def test_target_weak_and_strong_share_the_same_geometric_view(monkeypatch):
    transform = Compose([InitContentMask(), FixedSizeCrop((2, 2), random_crop=True), ToTensor()])
    dataset = _build_fake_stage_c_dataset(monkeypatch, transform, transform)

    random.seed(0)
    sample = dataset[0]

    assert torch.equal(sample["target_weak"]["image"], sample["target_strong"]["image"])
    assert torch.equal(sample["target_weak"]["depth"], sample["target_strong"]["depth"])
    assert torch.equal(sample["target_weak"]["content_mask"], sample["target_strong"]["content_mask"])
