from __future__ import annotations

import subprocess
from pathlib import Path


def test_smoke_matrix_dry_run_covers_requested_model_families(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260406_smoke_matrix.sh"

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--output-root",
            str(tmp_path / "smoke"),
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = res.stdout
    assert "scripts/create_fake_dataset.py" in stdout
    assert "magformer_depthnorm_on" in stdout
    assert "magformer_nodpth_ref_fair" in stdout
    assert "magformer_lightdepth_convnextlite_spatialgate_edge_validhole" in stdout
    assert "magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole" in stdout
    assert "magformer_lightdepth_mobilenetv3_sagate_edge_validhole" in stdout
    assert "run_official_mask2former_ecc.py" in stdout
    assert "run_detectron2_ecc.py" in stdout
    assert "train_net_mgm_0831.py" in stdout
    assert "run_msmformer_ecc.py" in stdout
    assert "run_ucn_ecc.py" in stdout
    assert "run_uoais_ecc.py" in stdout
    assert "run_unet_instance_ecc.py" in stdout
    assert "run_yolo_seg_ecc.py" in stdout
    assert "run_20260321_ddp_smoke_canary.sh" in stdout
    assert "--magformer-only" in stdout
    assert "--dry-run" in stdout


def test_smoke_matrix_dry_run_uses_true_short_budgets(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260406_smoke_matrix.sh"

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--output-root",
            str(tmp_path / "smoke"),
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = res.stdout
    assert "--max-iter 2" in stdout
    assert "--eval-period 2" in stdout
    assert "--checkpoint-period 2" in stdout
    assert "SOLVER.MAX_ITER 2" in stdout
    assert "TEST.EVAL_PERIOD 2" in stdout
    assert "--epochs 1" in stdout
