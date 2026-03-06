from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def _write_min_coco_instances(path: Path, num_images: int) -> None:
    images = [
        {"id": i + 1, "file_name": f"{i + 1:06d}.png", "width": 1024, "height": 1024}
        for i in range(num_images)
    ]
    data = {"images": images, "annotations": [], "categories": [{"id": 1, "name": "component"}]}
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    ("script_name", "variant", "expected_fragment"),
    [
        (
            "run_0831_1k_20ep_1024_revisit_magformer.sh",
            "nodpth_ref",
            "magformer_nodpth_ref",
        ),
        (
            "run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh",
            "depthnorm_on",
            "INPUT.DEPTH_PER_SAMPLE_NORM True",
        ),
    ],
)
def test_0831_1024_revisit_smoke_metadata_cmd_is_reproducible(
    tmp_path: Path,
    script_name: str,
    variant: str,
    expected_fragment: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / script_name
    assert script.exists()

    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", num_images=96)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", num_images=12)

    out_root = tmp_path / "out"
    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(out_root),
            "--variant",
            variant,
            "--smoke",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dry-run" in res.stdout
    assert "--smoke" in res.stdout
    assert expected_fragment in res.stdout

