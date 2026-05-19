from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from magformer.config import load_config

def _runtime_cfg(ema_enabled=False):
    return SimpleNamespace(
        ema_enabled=ema_enabled,
        resume=None,
        eval_period=10,
        checkpoint_period=10,
        log_period=1,
        find_unused_parameters=False,
        logger=SimpleNamespace(model_dump=lambda: {"type": "tensorboard"}),
    )


def _solver_cfg():
    return SimpleNamespace(
        ims_per_batch=2,
        amp_enabled=False,
        clip_gradients=True,
        clip_value=1.0,
    )


def _data_cfg():
    return SimpleNamespace(
        dataset_root="/tmp/vc_suda_dataset",
        train_ann="annotations/ordinary_train.json",
        val_ann="annotations/instances_val.json",
        train_split="train",
        val_split="val",
        image_size=32,
        min_scale=1.0,
        max_scale=1.0,
        random_flip="none",
        rgb_photo_aug=SimpleNamespace(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0),
        depth=SimpleNamespace(
            scale=1.0,
            shift=0.0,
            clip_min=0.0,
            clip_max=1.0,
            norm="minmax",
            per_sample_norm=True,
        ),
        depth_noise=SimpleNamespace(gaussian_std=0.0, speckle_std=0.0, drop_prob=0.0, drop_val=0.0),
    )


def _vc_suda_cfg(
    stage="C",
    source_ann="annotations/source_train.json",
    target_labeled_ann="annotations/target_labeled.json",
    target_unlabeled_ann="annotations/target_unlabeled.json",
    target_labeled_weight=1.0,
    source_root=None,
    source_datasets=None,
    offline_pseudo_enabled=False,
):
    offline_pseudo = SimpleNamespace(
        enabled=offline_pseudo_enabled,
        ann=(
            "output/diagnostics/r83_tta_coco_bank_20260517/instances_tta_pseudo_score090.json"
            if offline_pseudo_enabled
            else None
        ),
        quality_key="score",
        fill_ratio_key="fill_ratio",
        min_score=0.9,
        min_fill_ratio=0.0,
        max_instances=100,
        missing_image_policy="error",
        model_dump=lambda: {
            "enabled": offline_pseudo_enabled,
            "ann": (
                "output/diagnostics/r83_tta_coco_bank_20260517/instances_tta_pseudo_score090.json"
                if offline_pseudo_enabled
                else None
            ),
            "quality_key": "score",
            "fill_ratio_key": "fill_ratio",
            "min_score": 0.9,
            "min_fill_ratio": 0.0,
            "max_instances": 100,
            "missing_image_policy": "error",
        },
    )
    return SimpleNamespace(
        enabled=True,
        stage=stage,
        source_root=source_root,
        source_ann=source_ann,
        source_datasets=source_datasets,
        target_labeled_ann=target_labeled_ann,
        target_labeled_weight=target_labeled_weight,
        target_unlabeled_ann=target_unlabeled_ann,
        ema_teacher=SimpleNamespace(enabled=True, ema_momentum=0.999, warmup_steps=5),
        pseudo_label=SimpleNamespace(quality_threshold=0.5, use_curriculum=True, max_instances=7),
        curriculum=SimpleNamespace(start_threshold=0.7, end_threshold=0.3, warmup_epochs=2),
        domain_adaptation=SimpleNamespace(
            prototype_weight=0.0,
            boundary_weight=0.0,
            modality_dropout_weight=0.0,
            num_prototypes=4,
            boundary_confidence_threshold=0.5,
            modality_dropout_prob=0.0,
            use_uncertainty_weighting=False,
        ),
        unsupervised_weight=1.0,
        unsupervised_warmup_epochs=1,
        pseudo_exterior_ring_loss=SimpleNamespace(enabled=False, weight=0.0, radius=2),
        offline_pseudo=offline_pseudo,
        model_dump=lambda: {
            "enabled": True,
            "stage": stage,
            "source_root": source_root,
            "source_ann": source_ann,
            "target_labeled_ann": target_labeled_ann,
            "target_labeled_weight": target_labeled_weight,
            "target_unlabeled_ann": target_unlabeled_ann,
            "ema_teacher": {"enabled": True, "ema_momentum": 0.999, "warmup_steps": 5},
            "pseudo_label": {"quality_threshold": 0.5, "use_curriculum": True, "max_instances": 7},
            "curriculum": {"start_threshold": 0.7, "end_threshold": 0.3, "warmup_epochs": 2},
            "domain_adaptation": {
                "prototype_weight": 0.0,
                "boundary_weight": 0.0,
                "modality_dropout_weight": 0.0,
                "num_prototypes": 4,
                "boundary_confidence_threshold": 0.5,
                "modality_dropout_prob": 0.0,
                "use_uncertainty_weighting": False,
            },
            "unsupervised_weight": 1.0,
            "unsupervised_warmup_epochs": 1,
            "pseudo_exterior_ring_loss": {"enabled": False, "weight": 0.0, "radius": 2},
            "offline_pseudo": offline_pseudo.model_dump(),
        },
    )


def _config(
    stage="C",
    source_ann="annotations/source_train.json",
    target_labeled_ann="annotations/target_labeled.json",
    target_unlabeled_ann="annotations/target_unlabeled.json",
    target_labeled_weight=1.0,
    runtime_ema_enabled=False,
    source_root=None,
    source_datasets=None,
    offline_pseudo_enabled=False,
):
    return SimpleNamespace(
        data=_data_cfg(),
        solver=_solver_cfg(),
        runtime=_runtime_cfg(ema_enabled=runtime_ema_enabled),
        vc_suda=_vc_suda_cfg(
            stage=stage,
            source_ann=source_ann,
            target_labeled_ann=target_labeled_ann,
            target_unlabeled_ann=target_unlabeled_ann,
            target_labeled_weight=target_labeled_weight,
            source_root=source_root,
            source_datasets=source_datasets,
            offline_pseudo_enabled=offline_pseudo_enabled,
        ),
        model_dump=lambda: {"model": "dump"},
    )


def test_vc_suda_enabled_builds_semi_supervised_train_dataset(monkeypatch):
    from tools import train as train_tool
    import magformer.data as data_module
    import magformer.data.semi_supervised_dataset as semi_module

    calls = {}

    class FakeCocoDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSemiSupervisedDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls["semi_kwargs"] = kwargs

    monkeypatch.setattr(data_module, "CocoRgbdDataset", FakeCocoDataset)
    monkeypatch.setattr(semi_module, "SemiSupervisedDataset", FakeSemiSupervisedDataset)

    train_dataset, val_dataset = train_tool.build_datasets(_config(stage="C"))

    assert isinstance(train_dataset, FakeSemiSupervisedDataset)
    assert isinstance(val_dataset, FakeCocoDataset)
    assert calls["semi_kwargs"]["source_ann"] == "annotations/source_train.json"
    assert calls["semi_kwargs"]["target_unlabeled_ann"] == "annotations/target_unlabeled.json"
    assert calls["semi_kwargs"]["stage"] == "C"


def test_vc_suda_build_datasets_passes_target_unlabeled_sampling_stats(monkeypatch):
    from tools import train as train_tool
    import magformer.data as data_module
    import magformer.data.semi_supervised_dataset as semi_module

    calls = {}

    class FakeCocoDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSemiSupervisedDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls["semi_kwargs"] = kwargs

    monkeypatch.setattr(data_module, "CocoRgbdDataset", FakeCocoDataset)
    monkeypatch.setattr(semi_module, "SemiSupervisedDataset", FakeSemiSupervisedDataset)

    cfg = _config(stage="C")
    cfg.vc_suda.target_unlabeled_sampling = {
        "enabled": True,
        "stats_path": "output/diagnostics/r52/target_sampling_stats.json",
    }

    train_dataset, _ = train_tool.build_datasets(cfg)

    assert isinstance(train_dataset, FakeSemiSupervisedDataset)
    assert calls["semi_kwargs"]["target_unlabeled_sampling_stats"] == (
        "output/diagnostics/r52/target_sampling_stats.json"
    )


def test_vc_suda_build_datasets_passes_offline_pseudo_config(monkeypatch):
    from tools import train as train_tool
    import magformer.data as data_module
    import magformer.data.semi_supervised_dataset as semi_module

    calls = {}

    class FakeCocoDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSemiSupervisedDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls["semi_kwargs"] = kwargs

    monkeypatch.setattr(data_module, "CocoRgbdDataset", FakeCocoDataset)
    monkeypatch.setattr(semi_module, "SemiSupervisedDataset", FakeSemiSupervisedDataset)

    train_dataset, _ = train_tool.build_datasets(
        _config(stage="C", offline_pseudo_enabled=True)
    )

    assert isinstance(train_dataset, FakeSemiSupervisedDataset)
    assert calls["semi_kwargs"]["offline_pseudo_config"]["enabled"] is True
    assert calls["semi_kwargs"]["offline_pseudo_config"]["quality_key"] == "score"
    assert calls["semi_kwargs"]["offline_pseudo_config"]["min_score"] == pytest.approx(0.9)


def test_vc_suda_build_datasets_passes_multi_source_and_ignores_legacy_source(monkeypatch):
    from tools import train as train_tool
    import magformer.data as data_module
    import magformer.data.semi_supervised_dataset as semi_module

    calls = {}

    class FakeCocoDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSemiSupervisedDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls["semi_kwargs"] = kwargs

    monkeypatch.setattr(data_module, "CocoRgbdDataset", FakeCocoDataset)
    monkeypatch.setattr(semi_module, "SemiSupervisedDataset", FakeSemiSupervisedDataset)

    source_datasets = [
        {"name": "original", "root": "/data/original", "ann": "annotations/original.json", "weight": 1},
        {"name": "pseudo", "root": "/data/pseudo", "ann": "annotations/pseudo.json", "weight": 2},
    ]
    cfg = _config(stage="C", source_root="/legacy/source", source_datasets=source_datasets)

    train_dataset, _ = train_tool.build_datasets(cfg)

    assert isinstance(train_dataset, FakeSemiSupervisedDataset)
    assert calls["semi_kwargs"]["source_datasets"] == source_datasets
    assert calls["semi_kwargs"]["source_root"] is None
    assert calls["semi_kwargs"]["source_ann"] is None


def test_vc_suda_build_data_loaders_binds_transform_to_all_multi_source_datasets(monkeypatch):
    from tools import train as train_tool
    import magformer.data as data_module
    import magformer.data.semi_supervised_dataset as semi_module

    def numpy_sample(image_id):
        return {
            "image": np.full((4, 4, 3), image_id % 255, dtype=np.uint8),
            "depth": np.full((4, 4), float(image_id), dtype=np.float32),
            "image_id": image_id,
            "height": 4,
            "width": 4,
            "labels": np.array([1], dtype=np.int64),
            "masks": np.ones((4, 4, 1), dtype=bool),
            "boxes": np.array([[0, 0, 3, 3]], dtype=np.float32),
        }

    class FakeCocoDataset:
        samples_by_ann = {
            "annotations/source_a.json": [numpy_sample(100), numpy_sample(101)],
            "annotations/source_b.json": [numpy_sample(200), numpy_sample(201)],
            "annotations/instances_val.json": [numpy_sample(900)],
        }

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.transform = kwargs.get("transform")
            self.samples = list(self.samples_by_ann[kwargs["ann_file"]])
            self.image_ids = [sample["image_id"] for sample in self.samples]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = {
                key: value.copy() if hasattr(value, "copy") else value
                for key, value in self.samples[idx].items()
            }
            if self.transform is not None:
                sample = self.transform(sample)
            return sample

    monkeypatch.setattr(data_module, "CocoRgbdDataset", FakeCocoDataset)
    monkeypatch.setattr(semi_module, "CocoRgbdDataset", FakeCocoDataset)

    cfg = _config(
        stage="A",
        source_datasets=[
            {
                "name": "original",
                "root": "/data/source_a",
                "ann": "annotations/source_a.json",
                "weight": 1,
            },
            {
                "name": "pseudo",
                "root": "/data/source_b",
                "ann": "annotations/source_b.json",
                "weight": 1,
            },
        ],
    )

    train_dataset, val_dataset = train_tool.build_datasets(cfg)

    assert [dataset.transform for dataset in train_dataset.source_datasets] == [None, None]

    train_loader, _ = train_tool.build_data_loaders(
        cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=len(train_dataset),
        num_workers=0,
        is_distributed=False,
    )

    batch = next(iter(train_loader))

    assert set(batch["source_image_ids"].tolist()) == {100, 101, 200, 201}
    assert batch["source_images"].shape == (4, 3, 32, 32)
    assert all(
        dataset.transform is train_dataset.source.transform
        for dataset in train_dataset.source_datasets
    )


def test_vc_suda_build_data_loaders_uses_weak_augmentation_for_target_unlabeled(monkeypatch):
    from tools import train as train_tool
    import magformer.data.transforms as transforms_module
    from magformer.data.semi_supervised_dataset import SemiSupervisedDataset

    calls = {}
    strong_transform = SimpleNamespace(kind="strong")
    weak_transform = SimpleNamespace(kind="weak")

    def fake_weak_augmentation(data_cfg):
        calls["weak_data_cfg"] = data_cfg
        return weak_transform

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

    train_dataset = SemiSupervisedDataset.__new__(SemiSupervisedDataset)
    train_dataset.source = SimpleNamespace(transform=None)
    train_dataset.source_datasets = []
    train_dataset.target_labeled = SimpleNamespace(transform=None)
    train_dataset.weak_transform = None
    train_dataset.strong_transform = None
    train_dataset.set_source_transform = lambda transform: setattr(
        train_dataset.source,
        "transform",
        transform,
    )
    val_dataset = SimpleNamespace(transform=None)

    monkeypatch.setattr(transforms_module, "RGBDTransform", lambda **kwargs: strong_transform)
    monkeypatch.setattr(transforms_module, "get_weak_augmentation", fake_weak_augmentation)
    monkeypatch.setattr(train_tool, "DataLoader", FakeDataLoader)

    cfg = _config(stage="C")
    train_loader, _ = train_tool.build_data_loaders(
        cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=1,
        num_workers=0,
        is_distributed=False,
    )

    assert train_loader.dataset is train_dataset
    assert calls["weak_data_cfg"] is cfg.data
    assert train_dataset.weak_transform is weak_transform
    assert train_dataset.strong_transform is strong_transform
    assert train_dataset.weak_transform is not train_dataset.strong_transform


def test_vc_suda_enabled_requires_target_unlabeled_for_stage_c():
    from tools import train as train_tool

    with pytest.raises(ValueError, match="target_unlabeled_ann"):
        train_tool.validate_vc_suda_config(_config(stage="C", target_unlabeled_ann=None))


def test_vc_suda_stage_c_rejects_source_ann_reused_as_target_labeled():
    from tools import train as train_tool

    cfg = _config(
        stage="C",
        source_ann="annotations/shared_train.json",
        target_labeled_ann="annotations/shared_train.json",
    )

    with pytest.raises(ValueError, match="source_ann.*target_labeled_ann"):
        train_tool.validate_vc_suda_config(cfg)


@pytest.mark.parametrize("target_labeled_weight", [0.0, -0.1])
def test_vc_suda_stage_c_rejects_non_positive_target_labeled_weight(target_labeled_weight):
    from tools import train as train_tool

    cfg = _config(stage="C", target_labeled_weight=target_labeled_weight)

    with pytest.raises(ValueError, match="target_labeled_weight.*positive"):
        train_tool.validate_vc_suda_config(cfg)


def test_vc_suda_stage_c_rejects_generic_runtime_ema():
    from tools import train as train_tool

    with pytest.raises(ValueError, match="runtime.ema_enabled must be false"):
        train_tool.validate_vc_suda_config(_config(stage="C", runtime_ema_enabled=True))


def test_vc_suda_offline_pseudo_does_not_require_ema_teacher_config():
    from tools import train as train_tool

    cfg = _config(stage="C", offline_pseudo_enabled=True)
    cfg.vc_suda.ema_teacher = None

    train_tool.validate_vc_suda_config(cfg)


def test_vc_suda_enabled_builds_vc_suda_trainer(monkeypatch):
    from tools import train as train_tool

    captured = {}

    class FakePlainTrainer:
        def __init__(self, **kwargs):
            captured["plain"] = kwargs

    class FakeVCSUDATrainer:
        def __init__(self, **kwargs):
            captured["vc"] = kwargs

    model = SimpleNamespace(criterion=object())
    train_loader = object()
    val_loader = object()
    val_dataset = object()

    monkeypatch.setattr(train_tool, "Trainer", FakePlainTrainer)
    monkeypatch.setattr(train_tool, "VCSUDATrainer", FakeVCSUDATrainer, raising=False)

    cfg = _config(stage="C")
    cfg.vc_suda.pseudo_unmatched_negative_enabled = True
    cfg.vc_suda.pseudo_unmatched_negative_weight = 0.05
    cfg.vc_suda.pseudo_unmatched_negative_score_thresh = 0.9
    cfg.vc_suda.pseudo_exterior_ring_loss = SimpleNamespace(
        enabled=True,
        weight=0.05,
        radius=2,
    )

    trainer = train_tool.build_trainer(
        config=cfg,
        model=model,
        optimizer=object(),
        lr_scheduler=object(),
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        device=SimpleNamespace(type="cpu"),
        output_dir="output/test",
        is_distributed=False,
        amp_enabled=False,
    )

    assert isinstance(trainer, FakeVCSUDATrainer)
    assert "plain" not in captured
    assert captured["vc"]["train_loader"] is train_loader
    assert captured["vc"]["criterion"].supervised_criterion is model.criterion
    assert captured["vc"]["criterion"].pseudo_unmatched_negative_enabled is True
    assert captured["vc"]["criterion"].pseudo_unmatched_negative_weight == pytest.approx(0.05)
    assert captured["vc"]["criterion"].pseudo_unmatched_negative_score_thresh == pytest.approx(0.9)
    assert captured["vc"]["criterion"].pseudo_exterior_ring_enabled is True
    assert captured["vc"]["criterion"].pseudo_exterior_ring_weight == pytest.approx(0.05)
    assert captured["vc"]["criterion"].pseudo_exterior_ring_radius == 2
    assert captured["vc"]["ema_teacher"] is not None
    assert captured["vc"]["pseudo_label_scorer"] is not None
    assert captured["vc"]["curriculum_scheduler"] is not None



def test_vc_suda_offline_pseudo_builds_without_teacher_or_scorer(monkeypatch):
    from tools import train as train_tool

    captured = {}

    class FakeVCSUDATrainer:
        def __init__(self, **kwargs):
            captured["vc"] = kwargs

    model = SimpleNamespace(criterion=object())
    monkeypatch.setattr(train_tool, "VCSUDATrainer", FakeVCSUDATrainer, raising=False)

    cfg = _config(stage="C", offline_pseudo_enabled=True)

    train_tool.build_trainer(
        config=cfg,
        model=model,
        optimizer=object(),
        lr_scheduler=object(),
        train_loader=object(),
        val_loader=object(),
        val_dataset=object(),
        device=torch.device("cpu"),
        output_dir="output/test",
        is_distributed=False,
        amp_enabled=False,
    )

    assert captured["vc"]["ema_teacher"] is None
    assert captured["vc"]["pseudo_label_scorer"] is None
    assert captured["vc"]["curriculum_scheduler"] is not None


def test_vc_suda_stage_d_builds_required_domain_loss_modules(monkeypatch):
    from tools import train as train_tool

    captured = {}

    class FakeVCSUDATrainer:
        def __init__(self, **kwargs):
            captured["vc"] = kwargs

    cfg = _config(stage="D")
    cfg.vc_suda.domain_adaptation.prototype_weight = 1.0
    cfg.vc_suda.domain_adaptation.boundary_weight = 0.5
    cfg.vc_suda.domain_adaptation.modality_dropout_weight = 0.25

    model = SimpleNamespace(criterion=object())
    monkeypatch.setattr(train_tool, "VCSUDATrainer", FakeVCSUDATrainer, raising=False)

    train_tool.build_trainer(
        config=cfg,
        model=model,
        optimizer=object(),
        lr_scheduler=object(),
        train_loader=object(),
        val_loader=object(),
        val_dataset=object(),
        device=torch.device("cpu"),
        output_dir="output/test",
        is_distributed=False,
        amp_enabled=False,
    )

    assert set(captured["vc"]["domain_losses"]) == {"prototype", "boundary", "modality_dropout"}


def test_vc_suda_uncertainty_weighting_config_fails_fast():
    from tools import train as train_tool

    cfg = _config(stage="D")
    cfg.vc_suda.domain_adaptation.use_uncertainty_weighting = True

    with pytest.raises(ValueError, match="uncertainty weighting"):
        train_tool.validate_vc_suda_config(cfg)


def test_stage_a_file_routes_to_vc_suda_dataset_and_trainer(monkeypatch):
    from tools import train as train_tool
    import magformer.data as data_module
    import magformer.data.semi_supervised_dataset as semi_module

    cfg = load_config("configs/vc_suda_stage_a_40ep_512.yaml")

    assert cfg.vc_suda.enabled is True

    class FakeCocoDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSemiSupervisedDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeVCSUDATrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(data_module, "CocoRgbdDataset", FakeCocoDataset)
    monkeypatch.setattr(semi_module, "SemiSupervisedDataset", FakeSemiSupervisedDataset)
    monkeypatch.setattr(
        train_tool,
        "_build_vc_suda_components",
        lambda config, model, device: {"criterion": object()},
    )
    monkeypatch.setattr(train_tool, "VCSUDATrainer", FakeVCSUDATrainer, raising=False)

    train_dataset, val_dataset = train_tool.build_datasets(cfg)

    assert isinstance(train_dataset, FakeSemiSupervisedDataset)
    assert isinstance(val_dataset, FakeCocoDataset)
    assert train_dataset.kwargs["source_ann"] == "annotations/instances_source.json"
    assert train_dataset.kwargs["stage"] == "A"

    trainer = train_tool.build_trainer(
        config=cfg,
        model=SimpleNamespace(criterion=object()),
        optimizer=object(),
        lr_scheduler=object(),
        train_loader=object(),
        val_loader=object(),
        val_dataset=val_dataset,
        device=torch.device("cpu"),
        output_dir="output/test",
        is_distributed=False,
        amp_enabled=False,
    )

    assert isinstance(trainer, FakeVCSUDATrainer)


def test_stage_b_teacher8499_file_routes_source_and_target_labeled_dataset(monkeypatch):
    from tools import train as train_tool
    import magformer.data as data_module
    import magformer.data.semi_supervised_dataset as semi_module

    cfg = load_config("configs/vc_suda_stage_b_1024_teacher8499.yaml")

    class FakeCocoDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSemiSupervisedDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.source = SimpleNamespace(transform=None)
            self.target_labeled = SimpleNamespace(transform=None)

    monkeypatch.setattr(data_module, "CocoRgbdDataset", FakeCocoDataset)
    monkeypatch.setattr(semi_module, "SemiSupervisedDataset", FakeSemiSupervisedDataset)

    train_dataset, val_dataset = train_tool.build_datasets(cfg)

    assert isinstance(train_dataset, FakeSemiSupervisedDataset)
    assert isinstance(val_dataset, FakeCocoDataset)
    assert train_dataset.kwargs["stage"] == "B"
    assert train_dataset.kwargs["source_ann"] == "annotations/instances_source.json"
    assert train_dataset.kwargs["target_labeled_root"] == "magformer_datasets/pseudo_real_512"
    assert train_dataset.kwargs["target_labeled_ann"] == "annotations/instances_target_labeled.json"
    assert train_dataset.kwargs["target_unlabeled_ann"] == "annotations/instances_target_unlabeled.json"


HISTORICAL_INVALID_VC_SUDA_CONFIGS = [
    "configs/baseline_vc_suda_r118_magformer_r114warm_pseudo300.yaml",
    "configs/baseline_vc_suda_r121_depth_boundary_w001_smoke.yaml",
    "configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only.yaml",
    "configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.yaml",
    "configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml",
]


def test_historical_invalid_vc_suda_configs_fail_fast_for_collapsed_source_and_zero_target_weight():
    import yaml

    assert len(HISTORICAL_INVALID_VC_SUDA_CONFIGS) == 5

    for config_path in HISTORICAL_INVALID_VC_SUDA_CONFIGS:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        vc_suda = payload["vc_suda"]

        assert vc_suda["source_ann"] == vc_suda["target_labeled_ann"]
        assert vc_suda["target_labeled_weight"] == 0.0
        with pytest.raises(ValueError, match="source_ann|target_labeled_weight"):
            load_config(config_path)


def test_stage_c_r142_config_uses_32254_total_25654_source_train_split_online_ema():
    from tools import train as train_tool

    cfg = load_config(
        "configs/vc_suda_stage_c_r142_32254_train25654_source_target150_fixed.yaml"
    )

    train_tool.validate_vc_suda_config(cfg)

    assert cfg.name == "vc_suda_stage_c_r142_32254_train25654_source_target150_fixed"
    assert cfg.runtime.logger.run_name == (
        "vc_suda_stage_c_r142_32254_train25654_source_target150_fixed"
    )
    assert cfg.runtime.output_dir == (
        "output/vc_suda/stage_c_r142_32254_train25654_source_target150_fixed"
    )
    assert cfg.data.dataset_root == "magformer_datasets/pseudo_real_512"
    assert cfg.vc_suda.source_root == "magformer_datasets/20260318_1K_32254"
    assert cfg.vc_suda.source_ann == "cache/coco_loader/instances_train.sqlite"
    assert (
        f"{cfg.vc_suda.source_root}/{cfg.vc_suda.source_ann}"
        == "magformer_datasets/20260318_1K_32254/cache/coco_loader/instances_train.sqlite"
    )
    assert cfg.vc_suda.target_labeled_ann == (
        "annotations/instances_target_labeled_r114_balanced_plus125.json"
    )
    assert cfg.vc_suda.target_unlabeled_ann == (
        "annotations/instances_target_unlabeled_r114_balanced_minus125.json"
    )
    assert cfg.vc_suda.target_labeled_weight > 0
    assert cfg.vc_suda.offline_pseudo.enabled is False
    assert cfg.vc_suda.ema_teacher.enabled is True
    assert cfg.runtime.ema_enabled is False


def test_stage_b_r69_multisource_l2sp_config_routes_two_sources_and_retains_heads(monkeypatch):
    from tools import train as train_tool
    import magformer.data as data_module
    import magformer.data.semi_supervised_dataset as semi_module

    cfg = load_config("configs/vc_suda_stage_b_r69_multisource_l2sp_1024.yaml")

    assert cfg.name == "vc_suda_stage_b_r69_multisource_l2sp_1024"
    assert cfg.solver.max_iter == 500
    assert cfg.runtime.checkpoint_period == 500
    assert cfg.runtime.eval_period == 500
    assert cfg.runtime.resume is None
    assert cfg.model.finetune_weights == (
        "output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth"
    )
    assert getattr(cfg.model, "freeze_modules", None) in (None, [])
    assert cfg.vc_suda.source_retention.enabled is True
    assert cfg.vc_suda.source_retention.weight == pytest.approx(0.1)
    assert cfg.vc_suda.source_retention.normalize is True
    assert list(cfg.vc_suda.source_retention.include_prefixes) == ["decoder", "pixel_decoder"]
    assert list(cfg.vc_suda.source_retention.exclude_prefixes) == []

    class FakeCocoDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSemiSupervisedDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.source = SimpleNamespace(transform=None)
            self.target_labeled = SimpleNamespace(transform=None)

    monkeypatch.setattr(data_module, "CocoRgbdDataset", FakeCocoDataset)
    monkeypatch.setattr(semi_module, "SemiSupervisedDataset", FakeSemiSupervisedDataset)

    train_dataset, _ = train_tool.build_datasets(cfg)

    assert train_dataset.kwargs["stage"] == "B"
    assert train_dataset.kwargs["source_root"] is None
    assert train_dataset.kwargs["source_ann"] is None
    source_datasets = [source.model_dump() for source in train_dataset.kwargs["source_datasets"]]
    assert [source["name"] for source in source_datasets] == ["original", "pseudo"]
    assert [source["weight"] for source in source_datasets] == [1, 1]
    assert cfg.vc_suda.target_labeled_weight == pytest.approx(1.0)
    assert cfg.vc_suda.unsupervised_weight == pytest.approx(0.0)


def test_stage_b_r65_multisource_retention_config_routes_two_sources(monkeypatch):
    from tools import train as train_tool
    import magformer.data as data_module
    import magformer.data.semi_supervised_dataset as semi_module

    cfg = load_config("configs/vc_suda_stage_b_r65_multisource_retention_1024.yaml")

    class FakeCocoDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSemiSupervisedDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.source = SimpleNamespace(transform=None)
            self.target_labeled = SimpleNamespace(transform=None)

    monkeypatch.setattr(data_module, "CocoRgbdDataset", FakeCocoDataset)
    monkeypatch.setattr(semi_module, "SemiSupervisedDataset", FakeSemiSupervisedDataset)

    train_dataset, _ = train_tool.build_datasets(cfg)

    assert train_dataset.kwargs["stage"] == "B"
    assert train_dataset.kwargs["source_root"] is None
    assert train_dataset.kwargs["source_ann"] is None
    source_datasets = [source.model_dump() for source in train_dataset.kwargs["source_datasets"]]
    assert [source["name"] for source in source_datasets] == ["original", "pseudo"]
    assert [source["root"] for source in source_datasets] == [
        "magformer_datasets/20260318_1K_1566",
        "magformer_datasets/pseudo_real_512",
    ]
    assert [source["ann"] for source in source_datasets] == [
        "annotations/instances_train.json",
        "annotations/instances_source.json",
    ]
    assert [source["split"] for source in source_datasets] == ["train", "train"]
    assert [source["weight"] for source in source_datasets] == [1, 1]
    assert cfg.data.dataset_root == "magformer_datasets/pseudo_real_512"
    assert cfg.solver.base_lr == 1.0e-05
    assert cfg.solver.max_iter == 1000
    assert cfg.runtime.checkpoint_period == 500
    assert cfg.runtime.eval_period == 500
    assert cfg.vc_suda.target_labeled_weight == 1.0
    assert cfg.vc_suda.unsupervised_weight == 0.0
    assert cfg.vc_suda.ema_teacher.enabled is False

def test_vc_suda_stage_fields_do_not_enable_training_implicitly():
    from tools import train as train_tool

    cfg = load_config(
        "configs/vc_suda_stage_a_40ep_512.yaml",
        overrides={"vc_suda": {"enabled": False}},
    )

    assert cfg.vc_suda.enabled is False
    assert train_tool.is_vc_suda_enabled(cfg) is False


def test_vc_suda_source_root_overrides_only_source_dataset_root(monkeypatch):
    from tools import train as train_tool
    import magformer.data as data_module
    import magformer.data.semi_supervised_dataset as semi_module

    calls = {}

    class FakeCocoDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSemiSupervisedDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls["semi_kwargs"] = kwargs

    monkeypatch.setattr(data_module, "CocoRgbdDataset", FakeCocoDataset)
    monkeypatch.setattr(semi_module, "SemiSupervisedDataset", FakeSemiSupervisedDataset)

    train_tool.build_datasets(_config(stage="C", source_root="/data/source32k"))

    assert calls["semi_kwargs"]["source_root"] == "/data/source32k"
    assert calls["semi_kwargs"]["target_labeled_root"] == "/tmp/vc_suda_dataset"
    assert calls["semi_kwargs"]["target_unlabeled_root"] == "/tmp/vc_suda_dataset"
