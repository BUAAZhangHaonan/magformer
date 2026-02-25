from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _write_min_coco_instances(path: Path, num_images: int) -> None:
    images = [
        {"id": i + 1, "file_name": f"{i + 1:06d}.png", "width": 512, "height": 512}
        for i in range(num_images)
    ]
    data = {"images": images, "annotations": [], "categories": [{"id": 1, "name": "component"}]}
    path.write_text(json.dumps(data), encoding="utf-8")


def test_magformer_trackp_smoke_metadata_cmd_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_ecc_20ep_trackp_magformer.sh"
    assert script.exists()

    dataset_root = tmp_path / "0909_512_0.12K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", num_images=96)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", num_images=12)

    out_root = Path("out")

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "0909",
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

