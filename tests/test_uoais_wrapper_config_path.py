from __future__ import annotations

from pathlib import Path

from baselines.run_uoais_ecc import _ensure_dataset_name_overrides, _rewrite_passthrough_paths


def test_uoais_wrapper_rewrites_relative_config_file_to_repo_absolute() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rel = "configs/baselines/uoais_0831_1k_tracks.yaml"

    rewritten = _rewrite_passthrough_paths(["--config-file", rel], repo_root)

    assert rewritten[0] == "--config-file"
    assert rewritten[1] == str((repo_root / rel).resolve())


def test_uoais_wrapper_appends_dataset_name_overrides_when_missing() -> None:
    rewritten = _ensure_dataset_name_overrides(
        ["--num-gpus", "1"],
        train_name="ecc20260318_1k_1566_rgbd_train",
        val_name="ecc20260318_1k_1566_rgbd_val",
    )

    assert "DATASETS.TRAIN" in rewritten
    assert "DATASETS.TEST" in rewritten
    assert "('ecc20260318_1k_1566_rgbd_train',)" in rewritten
    assert "('ecc20260318_1k_1566_rgbd_val',)" in rewritten
