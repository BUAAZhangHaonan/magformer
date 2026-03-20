from __future__ import annotations

import json
import subprocess
import sys
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


def test_full19_roster_manifest_has_19_entries() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "full19_roster.py"

    res = subprocess.run(
        [sys.executable, str(script), "--format", "manifest"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(res.stdout)
    assert len(payload["models"]) == 19
    ids = {item["id"] for item in payload["models"]}
    assert "mask2former" in ids
    assert "msmformer" in ids
    assert "uoais" in ids
    assert "unet_boundary_inst" in ids


def test_full19_suite_dry_run_lists_19_models(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260321_20260318_1k_1566_full19.sh"
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_dataset(dataset_root)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert res.stdout.count(" START ") == 19
    for model_id in [
        "magformer_depthnorm_on",
        "mgm_mask2former_depthnorm_on",
        "mask2former",
        "maskrcnn",
        "yolov8_seg_x",
        "msmformer",
        "uoais",
        "unetpp_boundary_inst",
    ]:
        assert model_id in res.stdout
    assert "torchrun --nproc_per_node=2" in res.stdout
    assert "train_net_mgm_0831.py --num-gpus 2" in res.stdout
    assert "yolo segment train" in res.stdout
    assert "device=0,1" in res.stdout
