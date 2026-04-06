from __future__ import annotations

import subprocess
from pathlib import Path


def _ordered_positions(text: str, fragments: list[str]) -> list[int]:
    positions = []
    for fragment in fragments:
        pos = text.find(fragment)
        assert pos >= 0, fragment
        positions.append(pos)
    return positions


def test_gpu0_campaign_dry_run_matches_required_order_and_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260406_training_campaign_gpu0.sh"

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-base",
            str(tmp_path / "output"),
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = res.stdout
    positions = _ordered_positions(
        stdout,
        [
            "magformer_nodpth_ref_fair_1024",
            "magformer_depthnorm_on_512",
            "magformer_nodpth_ref_fair_512",
            "magformer_lightdepth_convnextlite_spatialgate_edge_validhole_512",
            "mgm_mask2former_depthnorm_on_512",
        ],
    )
    assert positions == sorted(positions)
    assert "CUDA_VISIBLE_DEVICES=0" in stdout
    assert "20260406_1k_1566_20ep_1024_full19" in stdout
    assert "20260406_1k_1566_20ep_512_full19" in stdout
    assert "--variant nodpth_ref_fair" in stdout
    assert "--variant depthnorm_on" in stdout
    assert "--variant convnextlite_spatialgate_edge_validhole" in stdout
    assert "--image-size 512" in stdout
    assert "wait_free_mb=78000" in stdout


def test_gpu1_campaign_dry_run_matches_required_order_and_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260406_training_campaign_gpu1.sh"

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-base",
            str(tmp_path / "output"),
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = res.stdout
    positions = _ordered_positions(
        stdout,
        [
            "official_mask2former_pretrained_512",
            "maskrcnn_pretrained_512",
            "mgm_mask2former_nodpth_ref_512",
            "yolov8_seg_x_pretrained_512",
        ],
    )
    assert positions == sorted(positions)
    assert "CUDA_VISIBLE_DEVICES=1" in stdout
    assert "--image-size 512" in stdout
    assert "--variant nodpth_ref" in stdout
    assert "--variant depthnorm_on" not in stdout
    assert "--model-size x" in stdout
    assert "--pretrained" in stdout
    assert "wait_free_mb=78000" in stdout
