from __future__ import annotations

import subprocess
from pathlib import Path


def test_baseline_repair_smoke_script_dry_run_covers_requested_model_families(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260406_baseline_repair_smokes.sh"

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--output-root",
            str(tmp_path / "smokes"),
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = res.stdout
    assert "create_fake_dataset.py" in stdout
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
    assert "--max-iter 2" in stdout
    assert "--epochs 1" in stdout
    assert "--magformer-only --run" in stdout


def test_baseline_repair_gpu_campaign_scripts_render_requested_order(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = repo_root / "magformer_datasets" / "20260318_1K_1566"
    gpu0_script = repo_root / "scripts" / "experiments" / "run_20260406_baseline_repair_gpu0.sh"
    gpu1_script = repo_root / "scripts" / "experiments" / "run_20260406_baseline_repair_gpu1.sh"

    gpu0 = subprocess.run(
        [
            "bash",
            str(gpu0_script),
            "--dataset-root",
            str(dataset_root),
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    gpu1 = subprocess.run(
        [
            "bash",
            str(gpu1_script),
            "--dataset-root",
            str(dataset_root),
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    gpu0_stdout = gpu0.stdout
    assert "run_0831_1k_20ep_1024_revisit_magformer.sh" in gpu0_stdout
    assert "--variant nodpth_ref_fair" in gpu0_stdout
    assert "--variant depthnorm_on" in gpu0_stdout
    assert "--image-size 512" in gpu0_stdout
    assert "run_0831_1k_20ep_1024_lightdepth_stage_a_magformer.sh" in gpu0_stdout
    assert "--variant convnextlite_spatialgate_edge_validhole" in gpu0_stdout
    assert "run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh" in gpu0_stdout
    assert "--output-root '/home/team/zhanghaonan/magformer/output/experiments/20260406_1k_1566_20ep_1024_full19'" in gpu0_stdout
    assert "--output-root '/home/team/zhanghaonan/magformer/output/experiments/20260406_1k_1566_20ep_512_full19'" in gpu0_stdout

    gpu1_stdout = gpu1.stdout
    assert "run_0831_1k_20ep_scratch_official_mask2former.sh" in gpu1_stdout
    assert "--pretrained" in gpu1_stdout
    assert "run_0831_1k_20ep_scratch_maskrcnn.sh" in gpu1_stdout
    assert "run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh" in gpu1_stdout
    assert "--variant nodpth_ref" in gpu1_stdout
    assert "run_0831_1k_20ep_scratch_yolov8_seg.sh" in gpu1_stdout
    assert "--model-size x" in gpu1_stdout
    assert "--image-size 512" in gpu1_stdout
