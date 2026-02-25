from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def _write_min_coco_instances(path: Path, num_images: int) -> None:
    images = [
        {"id": i + 1, "file_name": f"{i + 1:06d}.png", "width": 512, "height": 512}
        for i in range(num_images)
    ]
    data = {"images": images, "annotations": [], "categories": [{"id": 1, "name": "component"}]}
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    "script_name",
    [
        "run_0831_1k_20ep_scratch_magformer.sh",
        "run_0831_1k_20ep_scratch_mgm_mask2former.sh",
        "run_0831_1k_20ep_scratch_msmformer.sh",
        "run_0831_1k_20ep_scratch_uoais.sh",
        "run_0831_1k_20ep_scratch_ucn.sh",
        "run_0831_1k_20ep_scratch_official_mask2former.sh",
        "run_0831_1k_20ep_scratch_maskrcnn.sh",
        "run_0831_1k_20ep_scratch_yolov8_seg.sh",
    ],
)
def test_tracks_smoke_metadata_cmd_is_reproducible(tmp_path: Path, script_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / script_name
    assert script.exists()

    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", num_images=886)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", num_images=110)

    out_root = tmp_path / "out"

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(out_root),
            "--candidate-id",
            "C1",
            "--smoke",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    # `metadata.json` should store a runnable command. The runners support
    # --run/--dry-run and --smoke (flag), not `--mode` or `--smoke 0/1`.
    assert "--dry-run" in res.stdout
    assert "--smoke" in res.stdout
    assert "--smoke 1" not in res.stdout
    assert "--mode " not in res.stdout

