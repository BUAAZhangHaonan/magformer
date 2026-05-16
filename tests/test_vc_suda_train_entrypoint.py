from __future__ import annotations

from types import SimpleNamespace

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
    target_labeled_ann="annotations/target_labeled.json",
    target_unlabeled_ann="annotations/target_unlabeled.json",
    source_root=None,
    source_datasets=None,
):
    return SimpleNamespace(
        enabled=True,
        stage=stage,
        source_root=source_root,
        source_ann="annotations/source_train.json",
        source_datasets=source_datasets,
        target_labeled_ann=target_labeled_ann,
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
        model_dump=lambda: {
            "enabled": True,
            "stage": stage,
            "source_root": source_root,
            "source_ann": "annotations/source_train.json",
            "target_labeled_ann": target_labeled_ann,
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
        },
    )


def _config(
    stage="C",
    target_unlabeled_ann="annotations/target_unlabeled.json",
    runtime_ema_enabled=False,
    source_root=None,
    source_datasets=None,
):
    return SimpleNamespace(
        data=_data_cfg(),
        solver=_solver_cfg(),
        runtime=_runtime_cfg(ema_enabled=runtime_ema_enabled),
        vc_suda=_vc_suda_cfg(
            stage=stage,
            target_unlabeled_ann=target_unlabeled_ann,
            source_root=source_root,
            source_datasets=source_datasets,
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


def test_vc_suda_enabled_requires_target_unlabeled_for_stage_c():
    from tools import train as train_tool

    with pytest.raises(ValueError, match="target_unlabeled_ann"):
        train_tool.validate_vc_suda_config(_config(stage="C", target_unlabeled_ann=None))


def test_vc_suda_stage_c_rejects_generic_runtime_ema():
    from tools import train as train_tool

    with pytest.raises(ValueError, match="runtime.ema_enabled must be false"):
        train_tool.validate_vc_suda_config(_config(stage="C", runtime_ema_enabled=True))


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
