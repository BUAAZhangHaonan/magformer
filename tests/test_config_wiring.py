from magformer.config import load_config
from magformer.models import build_model


def _build(overrides=None):
    cfg = load_config(
        "configs/magformer_aligned_comparison.yaml",
        overrides={"data": {"dataset_root": "/tmp/dummy_dataset"}, **(overrides or {})},
    )
    return build_model(cfg)


def test_root_level_dpe_flag_is_honored_for_backward_compatibility():
    # Current aligned config uses legacy root-level dpe_enabled=true.
    model = _build()
    assert bool(model.pixel_decoder.dpe_enabled) is True


def test_nested_dpe_config_overrides_legacy_root_level_flag():
    model = _build(
        {
            "dpe_enabled": True,
            "model": {
                "magformer": {
                    "dpe": {
                        "enabled": False,
                        "beta": 7.0,
                    }
                }
            },
        }
    )
    assert bool(model.pixel_decoder.dpe_enabled) is False


def test_new_robust_norm_key_has_priority_over_legacy_prior_key():
    model = _build(
        {
            "model": {
                "magformer": {
                    "modality_fusion": {
                        "robust_norm_enabled": True,
                        "robust_norm_method": "minmax",
                        "prior": {
                            "robust_norm": False,
                            "robust_norm_method": "quantile",
                        },
                    }
                }
            }
        }
    )
    assert bool(model.fusion.prior_extractor.robust_norm) is True
    assert model.fusion.prior_extractor.robust_norm_method == "minmax"


def test_legacy_robust_norm_key_still_works_when_new_key_not_set():
    model = _build(
        {
            "model": {
                "magformer": {
                    "modality_fusion": {
                        "robust_norm_enabled": None,
                        "prior": {
                            "robust_norm": False,
                            "robust_norm_method": "quantile",
                        }
                    }
                }
            }
        }
    )
    assert bool(model.fusion.prior_extractor.robust_norm) is False
    assert model.fusion.prior_extractor.robust_norm_method == "quantile"
