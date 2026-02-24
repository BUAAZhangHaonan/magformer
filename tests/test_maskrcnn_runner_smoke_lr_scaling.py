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


def test_maskrcnn_trackp_smoke_scales_lr_with_batch(tmp_path: Path) -> None:
    """
    Smoke mode reduces batch size, so SOLVER.BASE_LR must be scaled down as well.
    Otherwise detectron2 training can diverge with Inf/NaN (observed on 0909).
    """

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_ecc_20ep_trackp_maskrcnn.sh"
    assert script.exists()

    dataset_root = tmp_path / "0909_512_0.12K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", num_images=96)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", num_images=12)

    out_root = tmp_path / "out"

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
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    # C1 is BASE_LR=0.01 for batch=8; smoke uses batch=2, so scaled lr should be 0.0025.
    assert "SOLVER.IMS_PER_BATCH 2" in res.stdout
    assert "SOLVER.BASE_LR 0.0025" in res.stdout

