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
    ("script_name", "expected_model_id", "expected_fragment"),
    [
        ("run_0831_1k_20ep_scratch_maskrcnn.sh", "maskrcnn_pretrained", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"),
        ("run_0831_1k_20ep_scratch_official_mask2former.sh", "official_mask2former_pretrained", "MODEL.WEIGHTS"),
        ("run_0831_1k_20ep_scratch_yolov8_seg.sh", "yolov8_seg_pretrained", "pretrained=True"),
    ],
)
def test_pretrained_runner_modes_have_distinct_output_ids(
    tmp_path: Path,
    script_name: str,
    expected_model_id: str,
    expected_fragment: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / script_name
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
            "--image-size",
            "1024",
            "--pretrained",
            "--smoke",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert expected_model_id in res.stdout
    assert expected_fragment in res.stdout
