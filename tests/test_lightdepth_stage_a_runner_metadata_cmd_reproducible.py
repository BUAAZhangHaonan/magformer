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


def test_lightdepth_stage_a_magformer_smoke_metadata_cmd_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_lightdepth_stage_a_magformer.sh"
    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", num_images=96)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", num_images=12)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            "out",
            "--variant",
            "mobilenetv3_directadd_edge",
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
    assert "--mode " not in res.stdout
    assert "model.magformer.modality_fusion.mode=direct_add" in res.stdout
    assert "model.magformer.modality_fusion.priors=[edge]" in res.stdout


def test_lightdepth_stage_a_magformer_supports_channel_and_spatial_variants(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_lightdepth_stage_a_magformer.sh"
    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", num_images=96)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", num_images=12)

    for variant, marker in [
        ("mobilenetv3_channelattn_edge", "model.magformer.modality_fusion.mode=channel_attn"),
        ("mobilenetv3_spatialgate_edge_validhole", "model.magformer.modality_fusion.mode=spatial_gate"),
        ("mobilenetv3_sagate_edge_validhole", "model.magformer.modality_fusion.mode=sa_gate"),
        ("mobilenetv3_esanetctx_edge_validhole", "model.magformer.modality_fusion.mode=esanet_ctx"),
    ]:
        res = subprocess.run(
            [
                "bash",
                str(script),
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                "out",
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
        assert marker in res.stdout
