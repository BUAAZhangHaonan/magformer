import copy
import subprocess
from pathlib import Path

import pytest

from magformer.config import load_config
from magformer.config.loader import load_yaml_file, save_yaml_file


STAGE_C_CONFIG = "configs/vc_suda_stage_c_1024_teacher8499.yaml"


def test_stage_c_config_has_safe_training_contract():
    cfg = load_config(STAGE_C_CONFIG)

    assert cfg.name == "vc_suda_stage_c_1024_teacher8499"
    assert cfg.vc_suda.enabled is True
    assert cfg.vc_suda.stage == "C"
    assert cfg.vc_suda.target_unlabeled_ann == "annotations/instances_target_unlabeled.json"
    assert cfg.vc_suda.ema_teacher.enabled is True
    assert cfg.vc_suda.unsupervised_weight > 0.0
    assert cfg.vc_suda.unsupervised_weight <= 0.1
    assert cfg.vc_suda.unsupervised_warmup_epochs >= 10
    assert set(cfg.runtime.eval_iou_types) == {"bbox", "segm"}
    assert cfg.runtime.resume is None
    assert cfg.model.finetune_weights
    assert "stage_b_1024_teacher8499" in cfg.model.finetune_weights
    assert cfg.model.finetune_weights.endswith("model_final.pth")
    assert cfg.data.depth.norm == "minmax"
    assert cfg.data.depth.per_sample_norm is True


def test_stage_c_static_preflight_passes_with_pending_stage_b_final_checkpoint():
    from tools.verify_vc_suda_stage import run_preflight

    result = run_preflight(
        STAGE_C_CONFIG,
        check_batch=False,
        require_finetune_exists=False,
    )

    assert result.config_path == Path(STAGE_C_CONFIG)
    assert "stage" in result.checks
    assert "checkpoint_semantics" in result.checks
    assert result.warnings == [
        "model.finetune_weights does not exist yet; allowed because Stage B final checkpoint is pending."
    ]


def test_stage_c_preflight_rejects_target_unlabeled_reusing_val(tmp_path):
    from tools.verify_vc_suda_stage import PreflightError, run_preflight

    raw = copy.deepcopy(load_yaml_file(STAGE_C_CONFIG))
    raw["vc_suda"]["target_unlabeled_ann"] = raw["data"]["val_ann"]
    bad_config = tmp_path / "bad_stage_c.yaml"
    save_yaml_file(raw, bad_config)

    with pytest.raises(PreflightError, match="target_unlabeled_ann.*val_ann"):
        run_preflight(bad_config, check_batch=False, require_finetune_exists=False)


def test_stage_c_preflight_rejects_resume_semantics(tmp_path):
    from tools.verify_vc_suda_stage import PreflightError, run_preflight

    raw = copy.deepcopy(load_yaml_file(STAGE_C_CONFIG))
    raw["runtime"]["resume"] = "output/experiments/stage_b/trainer_state_latest.pth"
    bad_config = tmp_path / "bad_resume.yaml"
    save_yaml_file(raw, bad_config)

    with pytest.raises(PreflightError, match="runtime.resume must be null"):
        run_preflight(bad_config, check_batch=False, require_finetune_exists=False)


def test_stage_c_preflight_cli_reports_pass_for_static_gate():
    completed = subprocess.run(
        [
            "python",
            "tools/verify_vc_suda_stage.py",
            "--config",
            STAGE_C_CONFIG,
            "--skip-batch",
            "--allow-missing-finetune",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "PASS config=configs/vc_suda_stage_c_1024_teacher8499.yaml" in completed.stdout
