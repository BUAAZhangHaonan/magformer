from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def _write_dataset(root: Path) -> None:
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "depth" / "depth_npy" / split).mkdir(parents=True, exist_ok=True)

        image_name = f"{split}_000001.png"
        Image.new("RGB", (16, 16), color=(24, 48, 96)).save(root / "images" / split / image_name)
        np.save(root / "depth" / "depth_npy" / split / f"{split}_000001.npy", np.full((16, 16), 0.5, dtype=np.float32))

        payload = {
            "images": [{"id": 1, "file_name": image_name, "width": 16, "height": 16}],
            "annotations": [],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_custom_register_maskrcnn_runner_uses_custom_dataset_names(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_maskrcnn.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_dataset(dataset_root)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--image-size",
            "1024",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ecc20260318_1k_1566_train" in res.stdout
    assert "ecc0831_1k_train" not in res.stdout
    assert "--register '20260318_1K_1566'" in res.stdout


def test_custom_register_uoais_runner_uses_custom_rgbd_dataset_names(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_uoais.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_dataset(dataset_root)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--image-size",
            "1024",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ecc20260318_1k_1566_rgbd_train" in res.stdout
    assert "ecc0831_1k_rgbd_train" not in res.stdout
    assert "--register '20260318_1K_1566'" in res.stdout


def test_custom_register_magformer_revisit_dry_run_writes_dynamic_stats_overrides(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_revisit_magformer.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_dataset(dataset_root)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--variant",
            "nodpth_ref",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--register '20260318_1K_1566'" in res.stdout
    assert "--dataset-root '" in res.stdout
    assert "render_magformer_runtime_config.py" in res.stdout
    assert "tools/evaluate.py" in res.stdout


def test_custom_register_lightdepth_stage_a_dry_run_writes_dynamic_stats_overrides(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_lightdepth_stage_a_magformer.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_dataset(dataset_root)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--variant",
            "mobilenetv3_spatialgate_edge_validhole",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--register '20260318_1K_1566'" in res.stdout
    assert "--dataset-root '" in res.stdout
    assert "render_magformer_runtime_config.py" in res.stdout


def test_custom_register_yolov8_pretrained_uses_managed_weights_and_dataset_stats(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_yolov8_seg.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_dataset(dataset_root)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--image-size",
            "1024",
            "--model-size",
            "s",
            "--pretrained",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "run_yolo_seg_ecc.py" in res.stdout
    assert "--model '" in res.stdout
    assert "--rgb-mean" in res.stdout
    assert "--rgb-std" in res.stdout


def test_custom_register_msmformer_runner_forces_unfrozen_backbone(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_msmformer.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_dataset(dataset_root)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MODEL.BACKBONE.FREEZE_AT 0" in res.stdout


def test_custom_register_unet_runner_passes_dataset_stats(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_revisit_unet_boundary_inst.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_dataset(dataset_root)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "20260318_1K_1566",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--rgb-mean" in res.stdout
    assert "--rgb-std" in res.stdout
    assert "--depth-clip-min" in res.stdout
    assert "--depth-clip-max" in res.stdout
