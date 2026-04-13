import sys
import types
import warnings

import torch
import torch.nn as nn


def _ensure_timm_stub() -> None:
    if "timm" in sys.modules:
        return

    class _FeatureInfo:
        def channels(self):
            return [64, 128, 256, 512]

        def reduction(self):
            return [4, 8, 16, 32]

    class _DummyTimmModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
            self.feature_info = _FeatureInfo()

    def _create_model(*args, **kwargs):
        del args, kwargs
        return _DummyTimmModel()

    timm = types.ModuleType("timm")
    timm.create_model = _create_model

    layers = types.ModuleType("timm.layers")

    class DropPath(nn.Identity):
        pass

    def to_2tuple(value):
        if isinstance(value, tuple):
            return value
        return (value, value)

    def trunc_normal_(tensor, std=0.02):
        del std
        return tensor

    layers.DropPath = DropPath
    layers.to_2tuple = to_2tuple
    layers.trunc_normal_ = trunc_normal_
    timm.layers = layers

    sys.modules["timm"] = timm
    sys.modules["timm.layers"] = layers


_ensure_timm_stub()

from magformer.config import load_config
from magformer.models import build_model


def _build(overrides=None):
    cfg = load_config(
        "configs/magformer_aligned_comparison.yaml",
        overrides={"data": {"dataset_root": "/tmp/dummy_dataset"}, **(overrides or {})},
    )
    return build_model(cfg)


def test_legacy_dpe_keys_emit_deprecation_warnings_and_map_to_nested_config():
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", DeprecationWarning)
        model = _build({"dpe_enabled": True, "dpe_beta": 7.0})

    messages = {str(item.message) for item in records}
    assert any("dpe_enabled" in message for message in messages)
    assert any("dpe_beta" in message for message in messages)
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


def test_pixel_decoder_ffn_defaults_to_1024_for_parity():
    model = _build()
    linear1 = model.pixel_decoder.transformer.encoder.layers[0].linear1
    assert int(linear1.weight.shape[0]) == 1024


def test_pixel_decoder_ffn_can_be_overridden_independently():
    model = _build(
        {
            "model": {
                "magformer": {
                    "sem_seg_head": {
                        "transformer_dim_feedforward": 1536,
                    },
                    "mask_former": {
                        "dim_feedforward": 2048,
                    },
                }
            }
        }
    )
    linear1 = model.pixel_decoder.transformer.encoder.layers[0].linear1
    assert int(linear1.weight.shape[0]) == 1536
