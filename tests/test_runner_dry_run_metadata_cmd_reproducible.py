from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def _write_min_coco_instances(path: Path, num_images: int, image_size: int) -> None:
    images = [
        {"id": i + 1, "file_name": f"{i + 1:06d}.png", "width": image_size, "height": image_size}
        for i in range(num_images)
    ]
    data = {"images": images, "annotations": [], "categories": [{"id": 1, "name": "component"}]}
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    "script_name",
    [
        "run_ecc_20ep_trackp_magformer.sh",
        "run_ecc_20ep_trackp_mgm_mask2former.sh",
        "run_ecc_20ep_trackp_msmformer.sh",
        "run_ecc_20ep_trackp_uoais.sh",
        "run_ecc_20ep_trackp_ucn.sh",
        "run_ecc_20ep_trackp_official_mask2former.sh",
        "run_ecc_20ep_trackp_maskrcnn.sh",
        "run_ecc_20ep_trackp_yolov8_seg.sh",
    ],
)
def test_trackp_runner_dry_run_metadata_cmd_is_reproducible(tmp_path: Path, script_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / script_name
    dataset_root = tmp_path / "0909_512_0.12K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", 96, 512)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", 12, 512)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--register",
            "0909",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--candidate-id",
            "C1",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dry-run" in res.stdout
    assert "--mode " not in res.stdout


@pytest.mark.parametrize(
    ("script_name", "extra_args", "expected_fragment"),
    [
        (
            "run_0831_1k_20ep_1024_revisit_magformer.sh",
            ["--variant", "nodpth_ref"],
            "magformer_nodpth_ref",
        ),
        (
            "run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh",
            ["--variant", "depthnorm_on"],
            "INPUT.DEPTH_PER_SAMPLE_NORM True",
        ),
        (
            "run_0831_1k_20ep_1024_revisit_unet_boundary_inst.sh",
            [],
            "unet_boundary_inst",
        ),
    ],
)
def test_0831_1024_revisit_runner_dry_run_metadata_cmd_is_reproducible(
    tmp_path: Path,
    script_name: str,
    extra_args: list[str],
    expected_fragment: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / script_name
    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", 96, 1024)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", 12, 1024)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            *extra_args,
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dry-run" in res.stdout
    assert expected_fragment in res.stdout


def test_lightdepth_stage_a_runner_dry_run_metadata_cmd_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_lightdepth_stage_a_magformer.sh"
    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", 96, 512)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", 12, 512)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--variant",
            "mobilenetv3_directadd_edge",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dry-run" in res.stdout
    assert "model.magformer.modality_fusion.mode=direct_add" in res.stdout
    assert "model.magformer.modality_fusion.priors=[edge]" in res.stdout


def test_lightdepth_stage_b_runner_dry_run_metadata_cmd_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_lightdepth_stage_b_magformer.sh"
    dataset_root = tmp_path / "0831_1K"
    (dataset_root / "annotations").mkdir(parents=True)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_train.json", 96, 512)
    _write_min_coco_instances(dataset_root / "annotations" / "instances_val.json", 12, 512)

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "out"),
            "--variant",
            "mobilenetv3_crossattn_edge_validhole",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dry-run" in res.stdout
    assert "model.magformer.modality_fusion.mode=cross_attn" in res.stdout
    assert "model.magformer.modality_fusion.priors=[edge,valid-hole]" in res.stdout
