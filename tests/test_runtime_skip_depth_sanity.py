from __future__ import annotations

from magformer.config import load_config


def test_runtime_skip_depth_sanity_can_be_overridden() -> None:
    cfg = load_config(
        "configs/baseline_supervised_r106_magformer_fulltarget200_oracle_warmstart_1000.yaml",
        overrides={
            "data": {"dataset_root": "/tmp/dummy_dataset"},
            "runtime": {"skip_depth_sanity": True},
        },
    )

    assert bool(cfg.runtime.skip_depth_sanity) is True


def test_runtime_depth_sanity_thresholds_can_be_configured() -> None:
    cfg = load_config(
        "configs/baseline_supervised_r106_magformer_fulltarget200_oracle_warmstart_1000.yaml",
        overrides={
            "data": {"dataset_root": "/tmp/dummy_dataset"},
            "runtime": {"depth_sanity": {"min_mask_fg_ratio": 0.0009}},
        },
    )

    assert cfg.runtime.depth_sanity.min_depth_range == 0.05
    assert cfg.runtime.depth_sanity.min_confidence_range == 1e-5
    assert cfg.runtime.depth_sanity.min_mask_fg_ratio == 0.0009
    assert cfg.runtime.depth_sanity.max_mask_fg_ratio == 0.999
    assert cfg.runtime.depth_sanity.depth_noise_sigma_multiplier == 6.0
