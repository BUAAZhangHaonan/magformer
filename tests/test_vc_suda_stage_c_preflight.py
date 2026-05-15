import copy
import subprocess
import sys
from pathlib import Path

import pytest

from magformer.config import load_config
from magformer.config.loader import load_yaml_file, save_yaml_file


STAGE_C_CONFIG = "configs/vc_suda_stage_c_1024_teacher8499.yaml"
STAGE_C_R3_A10_CONFIG = "configs/vc_suda_stage_c_r3_a10_1024_teacher8499.yaml"
STAGE_C_R6_NOCONTRAST_CONFIG = "configs/vc_suda_stage_c_r6_a10_nocontrast_1024_teacher8499.yaml"
STAGE_C_R7_LSJ10_CONFIG = "configs/vc_suda_stage_c_r7_a10_lsj10_1024_teacher8499.yaml"
STAGE_B_SEGM_EVAL_CONFIG = "configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml"


def test_stage_b_segm_post_eval_config_has_safe_contract():
    cfg = load_config(STAGE_B_SEGM_EVAL_CONFIG)

    assert cfg.name == "eval_vc_suda_stage_b_1024_teacher8499_segm"
    assert cfg.data.dataset_root == "magformer_datasets/pseudo_real_512"
    assert cfg.data.val_ann == "annotations/instances_val.json"
    assert cfg.data.val_split == "val"
    assert set(cfg.runtime.eval_iou_types) == {"bbox", "segm"}
    assert cfg.runtime.resume is None
    assert cfg.runtime.gpus == [0]
    assert cfg.runtime.ddp_enabled is False
    assert cfg.model.finetune_weights is None
    assert cfg.vc_suda.enabled is False
    assert cfg.vc_suda.target_labeled_ann in (None, "")
    assert cfg.vc_suda.target_unlabeled_ann in (None, "")


def test_stage_c_config_has_safe_training_contract():
    cfg = load_config(STAGE_C_CONFIG)

    assert cfg.name == "vc_suda_stage_c_1024_teacher8499"
    assert cfg.vc_suda.enabled is True
    assert cfg.vc_suda.stage == "C"
    assert cfg.vc_suda.target_unlabeled_ann == "annotations/instances_target_unlabeled.json"
    assert cfg.runtime.ema_enabled is False
    assert cfg.vc_suda.ema_teacher.enabled is True
    assert cfg.vc_suda.pseudo_label.quality_threshold == pytest.approx(0.2)
    assert cfg.vc_suda.pseudo_label.use_curriculum is True
    assert cfg.vc_suda.curriculum.start_threshold == pytest.approx(0.2)
    assert cfg.vc_suda.curriculum.end_threshold == pytest.approx(0.2)
    assert cfg.vc_suda.unsupervised_weight == pytest.approx(0.1)
    assert cfg.vc_suda.unsupervised_warmup_epochs >= 10
    assert cfg.runtime.eval_period == 1000
    assert cfg.runtime.eval_iou_types == ["bbox"]
    assert cfg.runtime.eval_max_images == 200
    assert cfg.runtime.eval_batch_size == 4
    assert cfg.runtime.resume is None
    assert cfg.model.finetune_weights
    assert "stage_b_1024_teacher8499" in cfg.model.finetune_weights
    assert cfg.model.finetune_weights.endswith("checkpoint_iter_0008999.pth")
    assert cfg.data.depth.norm == "minmax"
    assert cfg.data.depth.per_sample_norm is True


def test_stage_c_r6_nocontrast_config_has_single_variable_contract():
    cfg = load_config(STAGE_C_R6_NOCONTRAST_CONFIG)

    assert cfg.name == "vc_suda_stage_c_r6_a10_nocontrast_1024_teacher8499"
    assert cfg.runtime.output_dir == "output/vc_suda/stage_c_r6_a10_nocontrast_1024_teacher8499"
    assert cfg.runtime.logger.log_dir == "output/vc_suda/stage_c_r6_a10_nocontrast_1024_teacher8499/logs"
    assert cfg.runtime.logger.run_name == "vc_suda_stage_c_r6_a10_nocontrast_1024_teacher8499"
    assert cfg.model.finetune_weights == (
        "output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/"
        "checkpoint_iter_0008999.pth"
    )
    assert cfg.runtime.resume is None
    assert cfg.solver.max_iter == 2000
    assert cfg.runtime.checkpoint_period == 500
    assert cfg.runtime.eval_period == 1000
    assert cfg.runtime.eval_iou_types == ["bbox"]
    assert cfg.runtime.eval_max_images == 28
    assert cfg.runtime.eval_batch_size == 4
    assert cfg.runtime.contrastive_enabled is False
    assert cfg.vc_suda.target_labeled_weight == pytest.approx(1.0)
    assert cfg.vc_suda.pseudo_label.quality_threshold == pytest.approx(0.10)
    assert cfg.vc_suda.curriculum.start_threshold == pytest.approx(0.10)
    assert cfg.vc_suda.curriculum.end_threshold == pytest.approx(0.10)
    assert cfg.vc_suda.unsupervised_weight == pytest.approx(0.02)
    assert cfg.vc_suda.unsupervised_warmup_epochs == 10


def test_stage_c_r7_lsj10_config_only_fixes_lsj_scale_from_r3_a10():
    r3_raw = load_yaml_file(STAGE_C_R3_A10_CONFIG)
    r7_raw = load_yaml_file(STAGE_C_R7_LSJ10_CONFIG)

    expected = copy.deepcopy(r3_raw)
    expected["name"] = "vc_suda_stage_c_r7_a10_lsj10_1024_teacher8499"
    expected["data"]["min_scale"] = 1.0
    expected["data"]["max_scale"] = 1.0
    expected["solver"]["max_iter"] = 2000
    expected["runtime"]["output_dir"] = "output/vc_suda/stage_c_r7_a10_lsj10_1024_teacher8499"
    expected["runtime"]["checkpoint_period"] = 500
    expected["runtime"]["logger"]["log_dir"] = (
        "output/vc_suda/stage_c_r7_a10_lsj10_1024_teacher8499/logs"
    )
    expected["runtime"]["logger"]["run_name"] = "vc_suda_stage_c_r7_a10_lsj10_1024_teacher8499"

    assert r7_raw == expected

    cfg = load_config(STAGE_C_R7_LSJ10_CONFIG)
    assert cfg.runtime.contrastive_enabled is True
    assert cfg.data.min_scale == pytest.approx(1.0)
    assert cfg.data.max_scale == pytest.approx(1.0)
    assert cfg.vc_suda.target_labeled_weight == pytest.approx(1.0)
    assert cfg.vc_suda.pseudo_label.quality_threshold == pytest.approx(0.10)
    assert cfg.vc_suda.curriculum.start_threshold == pytest.approx(0.10)
    assert cfg.vc_suda.curriculum.end_threshold == pytest.approx(0.10)
    assert cfg.vc_suda.unsupervised_weight == pytest.approx(0.02)


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
    assert result.warnings in (
        [],
        ["model.finetune_weights does not exist yet; allowed because Stage B final checkpoint is pending."],
    )


def test_stage_c_preflight_allows_bbox_only_quick_eval():
    from tools.verify_vc_suda_stage import run_preflight

    result = run_preflight(
        STAGE_C_CONFIG,
        check_batch=False,
        require_finetune_exists=False,
    )

    assert "eval_iou_types" in result.checks
    assert "eval_max_images" in result.checks
    assert "eval_batch_size" in result.checks
    assert result.details["eval_iou_types"] == ["bbox"]
    assert result.details["eval_max_images"] == 200
    assert result.details["eval_batch_size"] == 4


def test_stage_c_preflight_rejects_unbounded_quick_eval_subset(tmp_path):
    from tools.verify_vc_suda_stage import PreflightError, run_preflight

    raw = copy.deepcopy(load_yaml_file(STAGE_C_CONFIG))
    raw["runtime"]["eval_max_images"] = 99999
    bad_config = tmp_path / "bad_eval_max_images.yaml"
    save_yaml_file(raw, bad_config)

    with pytest.raises(PreflightError, match="eval_max_images"):
        run_preflight(bad_config, check_batch=False, require_finetune_exists=False)


def test_stage_c_preflight_rejects_single_image_eval_batch_size(tmp_path):
    from tools.verify_vc_suda_stage import PreflightError, run_preflight

    raw = copy.deepcopy(load_yaml_file(STAGE_C_CONFIG))
    raw["runtime"]["eval_batch_size"] = 1
    bad_config = tmp_path / "bad_eval_batch_size.yaml"
    save_yaml_file(raw, bad_config)

    with pytest.raises(PreflightError, match="eval_batch_size"):
        run_preflight(bad_config, check_batch=False, require_finetune_exists=False)


def test_stage_c_preflight_rejects_model_final_placeholder(tmp_path):
    from tools.verify_vc_suda_stage import PreflightError, run_preflight

    raw = copy.deepcopy(load_yaml_file(STAGE_C_CONFIG))
    raw["model"]["finetune_weights"] = (
        "output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/model_final.pth"
    )
    bad_config = tmp_path / "bad_model_final.yaml"
    save_yaml_file(raw, bad_config)

    with pytest.raises(PreflightError, match="checkpoint_iter_0008999.pth"):
        run_preflight(bad_config, check_batch=False, require_finetune_exists=False)


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


def test_stage_c_preflight_rejects_generic_runtime_ema(tmp_path):
    from tools.verify_vc_suda_stage import PreflightError, run_preflight

    raw = copy.deepcopy(load_yaml_file(STAGE_C_CONFIG))
    raw["runtime"]["ema_enabled"] = True
    raw["vc_suda"]["ema_teacher"]["enabled"] = True
    bad_config = tmp_path / "bad_runtime_ema.yaml"
    save_yaml_file(raw, bad_config)

    with pytest.raises(PreflightError, match="runtime.ema_enabled must be false"):
        run_preflight(bad_config, check_batch=False, require_finetune_exists=False)


def test_stage_c_preflight_cli_reports_pass_for_static_gate():
    completed = subprocess.run(
        [
            sys.executable,
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
