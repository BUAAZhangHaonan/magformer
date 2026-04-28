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
            "iaunet_512",
            "cellpose_512",
            "stardist_512",
        ],
    )
    assert positions == sorted(positions)
    assert "CUDA_VISIBLE_DEVICES=1" in stdout
    assert "--image-size 512" in stdout
    assert "--variant nodpth_ref" in stdout
    assert "--variant depthnorm_on" not in stdout
    assert "--model-size x" in stdout
    assert "--pretrained" in stdout
    assert "run_0831_1k_20ep_1024_revisit_iaunet_inst.sh" in stdout
    assert "run_0831_1k_20ep_1024_revisit_cellpose_inst.sh" in stdout
    assert "run_0831_1k_20ep_1024_revisit_stardist_inst.sh" in stdout
    assert "wait_free_mb=78000" in stdout
    assert "min_ram_mb=50000" in stdout
    assert "max_swap_used_mb=1024" in stdout


def test_gpu0_finalize_and_continue_dry_run_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260407_finalize_fair_gpu0_and_continue.sh"

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--train-pid",
            "12345",
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
    assert "wait for pid 12345 to exit" in stdout
    assert "find_magformer_checkpoint.py" in stdout
    assert "write_params_from_magformer_ckpt.py" in stdout
    assert "tools/evaluate.py" in stdout
    assert "metrics.cocoeval.json" in stdout
    assert "run_20260406_training_campaign_gpu0.sh" in stdout


def test_gpu0_finalize_skip_wait_dry_run_still_resolves_best_checkpoint(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260407_finalize_fair_gpu0_and_continue.sh"

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--skip-wait",
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
    assert "skip wait and finalize immediately" in stdout
    assert "python3 compute wall_time and verify last_iter >= 6319" in stdout
    assert "find_magformer_checkpoint.py" in stdout
    assert "<best-magformer-checkpoint>" not in stdout


def test_gpu0_resume_fair_nohup_dry_run_uses_latest_checkpoint_and_finalize(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "output" / "20260406_1k_1566_20ep_1024_full19" / "magformer_nodpth_ref_fair"
    out_dir.mkdir(parents=True)
    (out_dir / "checkpoint_iter_0003159.pth").write_text("ckpt", encoding="utf-8")
    (out_dir / "magformer_runtime_config.yaml").write_text("name: test\n", encoding="utf-8")
    script = repo_root / "scripts" / "experiments" / "run_20260407_resume_fair_gpu0_nohup.sh"

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
    assert "checkpoint_iter_0003159.pth" in stdout
    assert "tools/train.py" in stdout
    assert "--resume" in stdout
    assert "run_20260407_finalize_fair_gpu0_and_continue.sh" in stdout


def test_gpu1_resume_mgm_and_continue_dry_run_is_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260407_resume_mgm_gpu1_and_continue.sh"

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
    assert "run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh" in stdout
    assert "--resume" in stdout
    assert "run_20260406_training_campaign_gpu1.sh" in stdout


def test_gpu1_non256_backfill_dry_run_matches_required_order_and_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260409_non256_completion_gpu1.sh"

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
            "magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole_512",
            "magformer_lightdepth_mobilenetv3_sagate_edge_validhole_512",
            "yolov8_seg_n_pretrained_512",
            "yolov8_seg_s_pretrained_512",
            "yolov8_seg_m_pretrained_512",
            "yolov8_seg_l_pretrained_512",
            "uoais_512",
            "unet_semantic_inst_512",
            "unet_boundary_inst_512",
            "unetpp_boundary_inst_512",
            "msmformer_512",
            "ucn_512",
            "iaunet_512",
            "cellpose_512",
            "stardist_512",
            "ucn_1024",
            "iaunet_1024",
            "cellpose_1024",
            "stardist_1024",
        ],
    )
    assert positions == sorted(positions)
    assert "CUDA_VISIBLE_DEVICES=1" in stdout
    assert "20260406_1k_1566_20ep_512_full19" in stdout
    assert "20260406_1k_1566_20ep_1024_full19" in stdout
    assert "--variant mobilenetv3_spatialgate_edge_validhole" in stdout
    assert "--variant mobilenetv3_sagate_edge_validhole" in stdout
    assert "--model-size n" in stdout
    assert "--model-size s" in stdout
    assert "--model-size m" in stdout
    assert "--model-size l" in stdout
    assert stdout.count("--image-size 512") >= 12
    assert "--image-size 1024" in stdout
    assert "run_0831_1k_20ep_1024_revisit_iaunet_inst.sh" in stdout
    assert "run_0831_1k_20ep_1024_revisit_cellpose_inst.sh" in stdout
    assert "run_0831_1k_20ep_1024_revisit_stardist_inst.sh" in stdout
    assert "wait_free_mb=78000" in stdout
    assert "min_ram_mb=50000" in stdout
    assert "max_swap_used_mb=1024" in stdout


def test_custom_unet_canary_dry_run_matches_required_order_and_controls(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260428_custom_unet_canaries_gpu1.sh"

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-base",
            str(tmp_path / "output"),
            "--max-train-steps",
            "50",
            "--max-val-images",
            "32",
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
            "cellpose_512_canary",
            "cellpose_1024_canary",
            "iaunet_512_canary",
            "iaunet_1024_canary",
        ],
    )
    assert positions == sorted(positions)
    assert "CUDA_VISIBLE_DEVICES=1" in stdout
    assert "20260428_custom_unet_canaries_gpu1" in stdout
    assert "--image-size 512" in stdout
    assert "--image-size 1024" in stdout
    assert "--batch 32" in stdout
    assert "--batch 16" in stdout
    assert "--batch 8" in stdout
    assert "--num-workers 4" in stdout
    assert "--num-queries 100" in stdout
    assert "--max-train-steps 50" in stdout
    assert "--max-val-images 32" in stdout
    assert "wait_free_mb=78000" in stdout
    assert "min_ram_mb=50000" in stdout
    assert "max_swap_used_mb=1024" in stdout


def test_custom_unet_full_dry_run_matches_required_order_and_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260428_custom_unet_full_gpu1.sh"

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
            "cellpose_512_full",
            "cellpose_1024_full",
            "iaunet_512_full",
            "iaunet_1024_full",
        ],
    )
    assert positions == sorted(positions)
    assert "CUDA_VISIBLE_DEVICES=1" in stdout
    assert "20260428_custom_unet_full_gpu1" in stdout
    assert "20260406_1k_1566_20ep_512_full19" in stdout
    assert "20260406_1k_1566_20ep_1024_full19" in stdout
    assert "run_0831_1k_20ep_1024_revisit_cellpose_inst.sh" in stdout
    assert "run_0831_1k_20ep_1024_revisit_iaunet_inst.sh" in stdout
    assert "--image-size 512" in stdout
    assert "--image-size 1024" in stdout
    assert "wait_free_mb=78000" in stdout
    assert "min_ram_mb=50000" in stdout
    assert "max_swap_used_mb=12000" in stdout


def test_custom_unet_postprocess_dry_run_waits_then_summarizes_visualizes_and_reports(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260429_custom_unet_postprocess_after_queue.sh"

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
            "wait_required_metrics",
            "summarize_512",
            "benchmark_512",
            "summarize_512_after_benchmark",
            "visualize_512",
            "summarize_1024",
            "benchmark_1024",
            "summarize_1024_after_benchmark",
            "visualize_1024",
            "build_live_manifest",
            "write_extended_metrics_table",
        ],
    )
    assert positions == sorted(positions)
    assert "visualize_suite.py" in stdout
    assert "summary_20260406_1k_1566_20ep_512_full19.json" in stdout
    assert "summary_20260406_1k_1566_20ep_1024_full19.json" in stdout
    assert "build_full19_live_metrics_manifest.py" in stdout
    assert "write_extended_metrics_table.py" in stdout


def test_gpu0_non256_backfill_dry_run_matches_required_order_and_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260409_non256_completion_gpu0.sh"

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
            "magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole_512",
            "magformer_lightdepth_mobilenetv3_sagate_edge_validhole_512",
            "yolov8_seg_n_pretrained_512",
            "yolov8_seg_s_pretrained_512",
            "yolov8_seg_m_pretrained_512",
            "yolov8_seg_l_pretrained_512",
            "uoais_512",
            "unet_semantic_inst_512",
            "unet_boundary_inst_512",
            "unetpp_boundary_inst_512",
            "msmformer_512",
            "ucn_512",
            "iaunet_512",
            "cellpose_512",
            "stardist_512",
            "ucn_1024",
            "iaunet_1024",
            "cellpose_1024",
            "stardist_1024",
        ],
    )
    assert positions == sorted(positions)
    assert "CUDA_VISIBLE_DEVICES=0" in stdout
    assert "20260406_1k_1566_20ep_512_full19" in stdout
    assert "20260406_1k_1566_20ep_1024_full19" in stdout
    assert "--variant mobilenetv3_spatialgate_edge_validhole" in stdout
    assert "--variant mobilenetv3_sagate_edge_validhole" in stdout
    assert "--model-size n" in stdout
    assert "--model-size s" in stdout
    assert "--model-size m" in stdout
    assert "--model-size l" in stdout
    assert stdout.count("--image-size 512") >= 12
    assert "--image-size 1024" in stdout
    assert "run_0831_1k_20ep_1024_revisit_iaunet_inst.sh" in stdout
    assert "run_0831_1k_20ep_1024_revisit_cellpose_inst.sh" in stdout
    assert "run_0831_1k_20ep_1024_revisit_stardist_inst.sh" in stdout
    assert "wait_free_mb=78000" in stdout


def test_gpu0_resume_non256_and_continue_dry_run_uses_latest_checkpoint_and_finalize(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (
        tmp_path
        / "output"
        / "20260406_1k_1566_20ep_512_full19"
        / "magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole"
    )
    out_dir.mkdir(parents=True)
    (out_dir / "checkpoint_iter_0000947.pth").write_text("ckpt-947", encoding="utf-8")
    (out_dir / "checkpoint_iter_0001263.pth").write_text("ckpt-1263", encoding="utf-8")
    (out_dir / "magformer_runtime_config.yaml").write_text("name: test\n", encoding="utf-8")
    script = repo_root / "scripts" / "experiments" / "run_20260409_resume_non256_gpu0_and_continue.sh"

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
    assert "checkpoint_iter_0001263.pth" in stdout
    assert "CUDA_VISIBLE_DEVICES=0 conda run -n magformer python tools/train.py" in stdout
    assert "--resume" in stdout
    assert "find_magformer_checkpoint.py" in stdout
    assert "tools/evaluate.py" in stdout
    assert "postprocess_cocoeval.py" in stdout
    assert "write_metrics_std.py" in stdout
    assert "prune_checkpoints.py" in stdout
    assert "run_20260409_non256_completion_gpu0.sh" in stdout
    assert "wait_free_mb=78000" in stdout


def test_gpu1_resume_msmformer_and_continue_dry_run_uses_resume_and_tail_queue(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "output" / "20260406_1k_1566_20ep_512_full19" / "msmformer"
    out_dir.mkdir(parents=True)
    (out_dir / "last_checkpoint").write_text("model_0000947.pth\n", encoding="utf-8")
    (out_dir / "model_0000947.pth").write_text("ckpt-947", encoding="utf-8")
    script = repo_root / "scripts" / "experiments" / "run_20260409_resume_msmformer_gpu1_and_continue.sh"

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
    assert "cuda_visible_devices=1" in stdout
    assert "last_checkpoint=" in stdout
    assert "baselines/run_msmformer_ecc.py" in stdout
    assert "--     --resume" in stdout
    assert "--num-gpus 1" in stdout
    assert "postprocess_cocoeval.py" in stdout
    assert "write_metrics_std.py" in stdout
    assert "prune_checkpoints.py" in stdout
    assert "write_run_metadata.py --phase end" in stdout
    assert "run_20260409_non256_completion_gpu1.sh" in stdout
    assert "wait_free_mb=78000" in stdout
