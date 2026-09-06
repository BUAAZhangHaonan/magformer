from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

from magformer.config import load_config


CONFIG_DIR = Path("configs/next_stage")
BASE_CONFIG = Path("configs/v317_init_from_m2f_swin_t.yaml")
MODEL_INIT = (
    "/home/g203-4028/magformer/output/experiments/"
    "v317_init_from_m2f/model_init.pth"
)
OUTPUT_ROOT = "/home/g203-4028/magformer/output/experiments/next_stage"
VALIDATED_TRAIN = "annotations/instances_train.validated.json"
VALIDATED_VAL = "annotations/instances_val.validated.json"
FOUR_LEVEL_CONFIG = CONFIG_DIR / "c0_corrected_4lvl_dpe_300k_seed42.yaml"
DPE_OFF_CONFIG = CONFIG_DIR / "dpe_off_corrected_300k_seed42.yaml"

CORRECTED_CONFIGS = {
    "c0_seed42": {
        "path": CONFIG_DIR / "c0_corrected_300k_seed42.yaml",
        "name": "20260717_next_stage_c0_corrected_300k_seed42",
        "output": f"{OUTPUT_ROOT}/c0_corrected_300k_seed42",
        "seed": 42,
        "confidence": True,
    },
    "c0_seed43": {
        "path": CONFIG_DIR / "c0_corrected_300k_seed43.yaml",
        "name": "20260725_next_stage_c0_corrected_300k_seed43",
        "output": f"{OUTPUT_ROOT}/c0_corrected_300k_seed43",
        "seed": 43,
        "confidence": True,
    },
    "d0_seed42": {
        "path": CONFIG_DIR / "d0_corrected_300k_seed42.yaml",
        "name": "20260725_next_stage_d0_corrected_300k_seed42",
        "output": f"{OUTPUT_ROOT}/d0_corrected_300k_seed42",
        "seed": 42,
        "confidence": False,
    },
}

EXPECTED = {
    "c0": {"loss": 12544, "oversample": 3.0, "confidence": True},
    "p1": {"loss": 25088, "oversample": 3.0, "confidence": True},
    "p2": {"loss": 12544, "oversample": 6.0, "confidence": True},
    "d0": {"loss": 12544, "oversample": 3.0, "confidence": False},
    "p3": {"loss": 50176, "oversample": 3.0, "confidence": True},
}

ALLOWED_V317_DIFFS = {
    ("name",),
    ("model", "magformer", "modality_fusion", "dccg_use_confidence"),
    ("model", "magformer", "modality_fusion", "dccg_conf_hidden"),
    ("model", "magformer", "modality_fusion", "residual_alpha"),
    ("model", "magformer", "modality_fusion", "noise_mask_weight"),
    ("model", "magformer", "modality_fusion", "hidden_dim"),
    ("model", "magformer", "modality_fusion", "feature_dims"),
    ("model", "magformer", "modality_fusion", "priors"),
    ("model", "magformer", "modality_fusion", "prior", "enabled"),
    ("model", "magformer", "modality_fusion", "prior", "use_gradient"),
    ("model", "magformer", "modality_fusion", "prior", "use_variance"),
    ("model", "magformer", "modality_fusion", "prior", "use_valid_hole"),
    ("model", "magformer", "modality_fusion", "prior", "use_rgb_edge"),
    ("model", "magformer", "modality_fusion", "prior", "var_kernel"),
    ("model", "magformer", "modality_fusion", "prior", "z_min"),
    ("model", "magformer", "modality_fusion", "prior", "z_max"),
    ("model", "magformer", "modality_fusion", "prior", "compute_on"),
    ("model", "magformer", "modality_fusion", "robust_norm_enabled"),
    ("model", "magformer", "modality_fusion", "robust_norm_method"),
    ("model", "magformer", "mask_former", "train_num_points"),
    ("model", "magformer", "mask_former", "matcher_num_points"),
    ("model", "magformer", "mask_former", "loss_num_points"),
    ("model", "magformer", "mask_former", "oversample_ratio"),
    ("solver", "max_iter"),
    ("runtime", "output_dir"),
    ("runtime", "cuda_memory_fraction"),
    ("runtime", "cpu_threads"),
    ("runtime", "cpu_interop_threads"),
    ("runtime", "amp_init_scale"),
    ("runtime", "max_consecutive_amp_skips"),
    ("runtime", "eval_period"),
    ("runtime", "checkpoint_period"),
    ("runtime", "resume"),
    ("runtime", "early_stop", "target_ap"),
    ("runtime", "early_stop", "enabled"),
    ("runtime", "early_stop", "monitor"),
    ("runtime", "early_stop", "target"),
    ("runtime", "early_stop", "min_optimizer_step"),
}

REQUIRED_V317_DIFFS = ALLOWED_V317_DIFFS - {
    ("model", "magformer", "mask_former", "oversample_ratio"),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict)
    return value


def _flatten(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        flattened: dict[tuple[str, ...], Any] = {}
        for key, child in value.items():
            flattened.update(_flatten(child, prefix + (str(key),)))
        return flattened
    return {prefix: value}


@pytest.mark.parametrize("case", sorted(CORRECTED_CONFIGS))
def test_corrected_configs_share_the_formal_300k_contract(case: str) -> None:
    expected = CORRECTED_CONFIGS[case]
    raw = _load_yaml(expected["path"])
    config = load_config(str(expected["path"]))

    assert config.name == expected["name"]
    assert config.data.train_ann == VALIDATED_TRAIN
    assert config.data.val_ann == VALIDATED_VAL
    assert config.model.finetune_weights == MODEL_INIT
    assert config.model.magformer is not None
    fusion = config.model.magformer.modality_fusion
    assert fusion.mode == "dccg"
    assert fusion.dccg_use_confidence is expected["confidence"]
    assert raw["model"]["magformer"]["modality_fusion"]["loss_entropy_w"] == 0.05
    assert config.solver.iteration_unit == "optimizer_step"
    assert config.solver.max_iter == 300000
    assert config.runtime.eval_period == 37500
    assert config.runtime.checkpoint_period == 37500
    assert config.runtime.resume is None
    assert config.runtime.output_dir == expected["output"]
    assert config.runtime.seed == expected["seed"]


@pytest.mark.parametrize(
    ("case", "expected_changes"),
    [
        (
            "c0_seed43",
            {
                ("name",),
                ("runtime", "output_dir"),
                ("runtime", "seed"),
            },
        ),
        (
            "d0_seed42",
            {
                ("name",),
                ("model", "magformer", "modality_fusion", "dccg_use_confidence"),
                ("runtime", "output_dir"),
            },
        ),
    ],
)
def test_corrected_variants_are_strict_single_variable_changes(
    case: str,
    expected_changes: set[tuple[str, ...]],
) -> None:
    control = _flatten(_load_yaml(CORRECTED_CONFIGS["c0_seed42"]["path"]))
    candidate = _flatten(_load_yaml(CORRECTED_CONFIGS[case]["path"]))
    changed = {
        path
        for path in set(control) | set(candidate)
        if control.get(path, object()) != candidate.get(path, object())
    }

    assert changed == expected_changes


def test_dpe_off_corrected_ablation_is_strictly_scoped() -> None:
    control_path = CORRECTED_CONFIGS["c0_seed42"]["path"]
    control_raw = _load_yaml(control_path)
    candidate_raw = _load_yaml(DPE_OFF_CONFIG)
    config = load_config(str(DPE_OFF_CONFIG))

    assert config.name == "20260725_next_stage_dpe_off_corrected_300k_seed42"
    assert config.data.train_ann == VALIDATED_TRAIN
    assert config.data.val_ann == VALIDATED_VAL
    assert config.model.finetune_weights == MODEL_INIT
    assert config.model.magformer is not None
    assert control_raw["model"]["magformer"]["dpe"]["enabled"] is True
    assert candidate_raw["model"]["magformer"]["dpe"]["enabled"] is False
    assert config.model.magformer.dpe.enabled is False
    assert config.solver.max_iter == 300000
    assert config.runtime.eval_period == 37500
    assert config.runtime.checkpoint_period == 37500
    assert config.runtime.resume is None
    assert config.runtime.seed == 42
    assert (
        config.runtime.output_dir
        == f"{OUTPUT_ROOT}/dpe_off_corrected_300k_seed42"
    )

    control = _flatten(control_raw)
    candidate = _flatten(candidate_raw)
    changed = {
        path
        for path in set(control) | set(candidate)
        if control.get(path, object()) != candidate.get(path, object())
    }
    assert changed == {
        ("name",),
        ("model", "magformer", "dpe", "enabled"),
        ("runtime", "output_dir"),
    }


def test_four_level_high_resolution_dpe_decoder_is_strictly_wired() -> None:
    control_path = CORRECTED_CONFIGS["c0_seed42"]["path"]
    raw = _load_yaml(FOUR_LEVEL_CONFIG)
    config = load_config(str(FOUR_LEVEL_CONFIG))

    assert (
        config.name
        == "20260725_next_stage_c0_corrected_4_level_high_resolution_dpe_decoder_300k_seed42"
    )
    assert config.data.train_ann == VALIDATED_TRAIN
    assert config.data.val_ann == VALIDATED_VAL
    assert config.model.finetune_weights == MODEL_INIT
    assert config.model.magformer is not None
    assert config.model.magformer.mask_former.num_feature_levels == 4
    assert (
        config.model.magformer.sem_seg_head.deformable_transformer_encoder_in_features
        == ["res2", "res3", "res4", "res5"]
    )
    assert config.solver.max_iter == 300000
    assert config.runtime.eval_period == 37500
    assert config.runtime.checkpoint_period == 37500
    assert config.runtime.resume is None
    assert (
        config.runtime.output_dir
        == f"{OUTPUT_ROOT}/c0_corrected_4lvl_dpe_300k_seed42"
    )

    control = _flatten(_load_yaml(control_path))
    candidate = _flatten(raw)
    changed = {
        path
        for path in set(control) | set(candidate)
        if control.get(path, object()) != candidate.get(path, object())
    }
    assert changed == {
        ("name",),
        ("model", "magformer", "mask_former", "num_feature_levels"),
        (
            "model",
            "magformer",
            "sem_seg_head",
            "deformable_transformer_encoder_in_features",
        ),
        ("runtime", "output_dir"),
    }


@pytest.mark.parametrize("arm", sorted(EXPECTED))
def test_next_stage_config_matches_declared_matrix(arm: str) -> None:
    path = CONFIG_DIR / f"{arm}_seed42.yaml"
    raw = _load_yaml(path)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        config = load_config(str(path))

    assert not any("train_num_points is deprecated" in str(item.message) for item in records)
    assert not any("target_ap is deprecated" in str(item.message) for item in records)

    expected = EXPECTED[arm]
    magformer = config.model.magformer
    assert magformer is not None
    mask_former = magformer.mask_former
    fusion = magformer.modality_fusion
    raw_fusion = raw["model"]["magformer"]["modality_fusion"]
    assert set(raw_fusion) == {
        "enabled",
        "mode",
        "dccg_use_confidence",
        "dccg_conf_hidden",
        "loss_entropy_w",
        "temp_init",
        "temp_final",
        "temp_steps",
        "clamp_min",
        "clamp_max",
        "scale_keys",
        "fuse_scales",
        "priors",
        "post_fuse_norm",
        "prior",
    }
    assert raw_fusion["priors"] == []
    assert raw_fusion["prior"] == {"enabled": False}
    assert fusion.dccg_conf_hidden == 16
    assert fusion.prior.enabled is False

    assert config.name == f"20260712_next_stage_{arm}_seed42"
    assert config.model.finetune_weights == MODEL_INIT
    assert magformer.modality_fusion.mode == "dccg"
    assert magformer.modality_fusion.dccg_use_confidence is expected["confidence"]
    assert mask_former.matcher_num_points == 12544
    assert mask_former.loss_num_points == expected["loss"]
    assert mask_former.oversample_ratio == pytest.approx(expected["oversample"])
    assert mask_former.importance_sample_ratio == pytest.approx(0.75)
    assert "train_num_points" not in raw["model"]["magformer"]["mask_former"]

    assert config.solver.max_iter == 128000
    assert config.solver.warmup_iters == 1000
    assert config.runtime.grad_accum_steps == 4
    raw_runtime = raw["runtime"]
    assert config.runtime.cuda_memory_fraction == 0.85
    assert config.runtime.cpu_threads == 8
    assert config.runtime.cpu_interop_threads == 1
    assert raw_runtime["cuda_memory_fraction"] == 0.85
    assert raw_runtime["cpu_threads"] == 8
    assert raw_runtime["cpu_interop_threads"] == 1
    assert config.runtime.amp_init_scale == 64.0
    assert config.runtime.max_consecutive_amp_skips == 16
    assert raw_runtime["amp_init_scale"] == 64.0
    assert type(raw_runtime["amp_init_scale"]) is float
    assert raw_runtime["max_consecutive_amp_skips"] == 16
    assert type(raw_runtime["max_consecutive_amp_skips"]) is int
    assert config.runtime.eval_period == 16000
    assert config.runtime.checkpoint_period == 16000
    assert config.runtime.seed == 42
    assert config.runtime.resume is None
    assert config.runtime.output_dir == f"{OUTPUT_ROOT}/{arm}_seed42"
    assert config.runtime.early_stop.model_dump() == {
        "enabled": False,
        "monitor": "val/mAP",
        "patience": 6,
        "min_delta": 0.001,
        "target": 0.9,
        "min_optimizer_step": 0,
    }


@pytest.mark.parametrize("arm", sorted(EXPECTED))
def test_next_stage_config_only_changes_approved_v317_fields(arm: str) -> None:
    base = _flatten(_load_yaml(BASE_CONFIG))
    candidate = _flatten(_load_yaml(CONFIG_DIR / f"{arm}_seed42.yaml"))
    all_paths = set(base) | set(candidate)
    changed = {
        path
        for path in all_paths
        if base.get(path, object()) != candidate.get(path, object())
    }

    assert changed <= ALLOWED_V317_DIFFS
    assert REQUIRED_V317_DIFFS <= changed
    if arm == "p2":
        assert ("model", "magformer", "mask_former", "oversample_ratio") in changed
    else:
        assert ("model", "magformer", "mask_former", "oversample_ratio") not in changed


def test_promotion_documentation_is_non_runnable_and_covers_both_seeds() -> None:
    text = (CONFIG_DIR / "PROMOTION.md").read_text(encoding="utf-8")

    assert "cross-file inheritance" in text
    assert "P3 is conditional" in text
    assert "<arm>_seed43.yaml" in text
    assert "<arm>_seed44.yaml" in text
    assert "runtime.resume: null" in text


def test_readme_describes_confidence_only_dccg_scope() -> None:
    text = (CONFIG_DIR / "README.md").read_text(encoding="utf-8")

    assert "bidirectional cross-attention path" in text
    assert "without learned depth confidence" in text
    assert "valid-hole" in text
    assert "`residual_alpha`" in text
    assert "They do not treat" in text
    assert "prior:\n  enabled: false" in text
    assert "output/smoke/20260712_062711_4512c35_scale_diag/R64_summary.json" in text
    assert "output/smoke/20260712_062711_4512c35_scale_diag/R1_summary.json" in text
    assert "output/smoke/20260712_062711_4512c35_scale_diag/scale_comparison.json" in text
    assert "R64: PASS" in text
    assert "R1: PASS" in text
    assert "0.6818156121" in text
    assert "0.6817464737" in text
    assert "Both R64 and R1 pass" in text
    assert "`amp_init_scale: 64.0`" in text
    assert "underflow headroom" in text
    assert "`max_consecutive_amp_skips: 16`" in text
