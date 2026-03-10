from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _write_min_coco_instances(path: Path, num_images: int) -> None:
    images = [
        {"id": i + 1, "file_name": f"{i + 1:06d}.png", "width": 1024, "height": 1024}
        for i in range(num_images)
    ]
    data = {"images": images, "annotations": [], "categories": [{"id": 1, "name": "component"}]}
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_reference_bank(ref_root: Path) -> None:
    for sub in ["rgb", "depth", "mask"]:
        (ref_root / sub).mkdir(parents=True)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = (30, 50, 70)
    cv2.imwrite(str(ref_root / "rgb" / "view0.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    np.save(ref_root / "depth" / "view0.npy", np.full((32, 32), 0.95, dtype=np.float32))
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(ref_root / "mask" / "view0.png"), mask)


def test_unet_reference_runner_smoke_metadata(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_revisit_unet_reference_inst.sh"
    assert script.exists()

    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", num_images=96)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", num_images=12)
    ref_root = tmp_path / "refs"
    _write_reference_bank(ref_root)

    out_root = tmp_path / "out"
    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(out_root),
            "--reference-root",
            str(ref_root),
            "--smoke",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--reference-root" in res.stdout
    assert "--variant 'unet_reference_inst'" in res.stdout
