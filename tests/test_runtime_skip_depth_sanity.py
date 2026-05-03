from __future__ import annotations

from magformer.config import load_config


def test_runtime_skip_depth_sanity_can_be_overridden() -> None:
    cfg = load_config(
        "configs/magformer_0831_1k_20ep_1024_nodpth_ref.yaml",
        overrides={
            "data": {"dataset_root": "/tmp/dummy_dataset"},
            "runtime": {"skip_depth_sanity": True},
        },
    )

    assert bool(cfg.runtime.skip_depth_sanity) is True
