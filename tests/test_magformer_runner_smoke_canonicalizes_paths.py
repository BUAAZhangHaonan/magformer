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


def test_magformer_trackp_smoke_canonicalizes_out_dir(tmp_path: Path) -> None:
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

    out_dir = (tmp_path / out_root / "magformer").as_posix()
    assert f"--out-dir '{out_dir}'" in res.stdout
    assert f"--output-dir '{out_dir}'" in res.stdout
    assert f"--out-config '{out_dir}/magformer_runtime_config.yaml'" in res.stdout
    assert f"tools/train.py --config '{out_dir}/magformer_runtime_config.yaml'" in res.stdout
