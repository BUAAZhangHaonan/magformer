import subprocess

from pathlib import Path


SCRIPT = Path("tools/start_vc_suda_watchers.sh")
CURRENT_R142_CONFIG = "configs/vc_suda_stage_c_r142_32254_train25654_source_target150_fixed.yaml"
HISTORICAL_INVALID_CONFIGS = [
    "configs/baseline_vc_suda_r118_magformer_r114warm_pseudo300.yaml",
    "configs/baseline_vc_suda_r121_depth_boundary_w001_smoke.yaml",
    "configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only.yaml",
    "configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.yaml",
    "configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml",
]


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_start_script_is_retired_noop_and_isolates_historical_invalid_configs() -> None:
    text = _script_text()

    for config_path in HISTORICAL_INVALID_CONFIGS:
        assert config_path not in text

    assert "retired" in text.lower()
    assert "no-op" in text.lower()
    assert "historical invalid" in text.lower()
    assert "source/target collapse" in text
    assert "target_labeled_weight=0" in text


def test_start_script_points_current_stage_c_note_at_r142_positive_config() -> None:
    text = _script_text()

    assert CURRENT_R142_CONFIG in text
    assert "32254_train25654" in text
    assert "magformer_datasets/20260318_1K_32254/cache/coco_loader/instances_train.sqlite" in text


def test_start_script_does_not_launch_training_or_eval_jobs() -> None:
    text = _script_text()

    forbidden_fragments = [
        "baseline_vc_suda_r118",
        "baseline_vc_suda_r121",
        "baseline_vc_suda_r122",
        "torch.distributed.run",
        "tools/train.py",
        "tools/evaluate_1024_backmap.py",
        "CUDA_VISIBLE_DEVICES=",
        "tmux new-session",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in text


def test_start_script_reports_historical_watchers_are_disabled_without_gpu_probe() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "retired" in output.lower()
    assert "no-op" in output.lower()
    assert "historical invalid" in output.lower()
    assert CURRENT_R142_CONFIG in output
    assert "cuda_available" not in output
