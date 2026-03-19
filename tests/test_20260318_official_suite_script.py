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


def test_20260318_official_suite_dry_run_has_26_entries_and_excludes_ucn(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260318_1k_1566_20ep_1024_official_all.sh"
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

    assert res.stdout.count(" START ") == 26
    assert "ucn" not in res.stdout.lower()
    assert "magformer_depthnorm_on" in res.stdout
    assert "mgm_mask2former_depthnorm_on" in res.stdout
    assert "official_mask2former_pretrained" in res.stdout
    assert "maskrcnn_pretrained" in res.stdout
    assert "msmformer_scratch" in res.stdout
    assert "uoais_scratch" in res.stdout
    assert "unet_boundary_inst" in res.stdout
    assert "unetpp_boundary_inst" in res.stdout
    assert "yolov8_seg_x_pretrained" in res.stdout
    assert "yolov8_seg_x_scratch" in res.stdout
