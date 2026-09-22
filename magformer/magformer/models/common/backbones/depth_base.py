# -*- coding: utf-8 -*-
"""
Shared utilities for timm-based depth backbones.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ....engine.utils import load_torch_checkpoint


_CANONICAL_REDUCTIONS = {
    "res2": 4,
    "res3": 8,
    "res4": 16,
    "res5": 32,
}


def select_timm_out_indices(
    reductions: Sequence[int],
    out_features: Sequence[str],
) -> Tuple[int, ...]:
    reduction_list = [int(value) for value in reductions]
    indices: List[int] = []
    for name in out_features:
        if name not in _CANONICAL_REDUCTIONS:
            raise KeyError(f"Unsupported canonical feature name: {name}")
        target_reduction = _CANONICAL_REDUCTIONS[name]
        try:
            indices.append(reduction_list.index(target_reduction))
        except ValueError as exc:
            raise ValueError(
                f"Could not map feature '{name}' (reduction={target_reduction}) "
                f"from available reductions {reduction_list}"
            ) from exc
    return tuple(indices)


class TimmDepthBackboneBase(nn.Module):
    """
    Common implementation for timm-based single-channel depth backbones.
    """

    def __init__(
        self,
        *,
        model_name: str,
        out_features: Sequence[str],
        pretrained: bool = False,
        weights_path: Optional[str] = None,
        strict_weights: bool = False,
    ) -> None:
        super().__init__()
        import timm

        self.model_name = str(model_name)
        self.out_features = list(out_features)
        self.strict_weights = bool(strict_weights)
        self.pretrained_load_report: Optional[Dict[str, object]] = None

        probe = timm.create_model(
            self.model_name,
            pretrained=pretrained and (weights_path is None),
            features_only=True,
            in_chans=1,
        )
        feature_info = probe.feature_info
        all_reductions = [int(x) for x in feature_info.reduction()]
        all_channels = [int(x) for x in feature_info.channels()]
        out_indices = select_timm_out_indices(
            all_reductions, self.out_features)
        del probe

        self.model = timm.create_model(
            self.model_name,
            pretrained=pretrained and (weights_path is None),
            features_only=True,
            out_indices=out_indices,
            in_chans=1,
        )

        if weights_path is not None:
            self._load_weights(weights_path)

        self._stage_out_channels = {
            name: all_channels[idx] for name, idx in zip(self.out_features, out_indices)
        }
        self._stage_out_strides = {
            name: all_reductions[idx] for name, idx in zip(self.out_features, out_indices)
        }

    def _load_weights(self, weights_path: str) -> None:
        state_dict = load_torch_checkpoint(weights_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model_state = self.model.state_dict()
        filtered_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and value.shape == model_state[key].shape
        }
        shape_mismatches = sorted(
            key
            for key, value in state_dict.items()
            if key in model_state and value.shape != model_state[key].shape
        )
        unexpected = sorted(key for key in state_dict if key not in model_state)
        missing = sorted(key for key in model_state if key not in filtered_state)
        self.pretrained_load_report = {
            "checkpoint": str(weights_path),
            "matched": len(filtered_state),
            "expected": len(model_state),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatch_keys": shape_mismatches,
        }
        if self.strict_weights and (missing or unexpected or shape_mismatches):
            raise RuntimeError(
                "Depth checkpoint is not an exact match: "
                f"matched={len(filtered_state)}/{len(model_state)}, "
                f"missing={missing}, unexpected={unexpected}, "
                f"shape_mismatches={shape_mismatches}"
            )
        incompatible = self.model.load_state_dict(
            filtered_state,
            strict=self.strict_weights,
        )
        if not self.strict_weights:
            print(
                "[DepthBackbone] Explicit non-strict load report: "
                f"matched={len(filtered_state)}/{len(model_state)}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}, "
                f"shape_mismatches={len(shape_mismatches)}"
            )
            if sorted(incompatible.missing_keys) != missing:
                raise RuntimeError(
                    "Depth checkpoint load report disagrees with PyTorch "
                    f"missing keys: computed={missing}, "
                    f"pytorch={incompatible.missing_keys}"
                )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.model(x)
        result: Dict[str, torch.Tensor] = {}
        for idx, feat in enumerate(features):
            stage_name = self.out_features[idx]
            expected_ch = self._stage_out_channels[stage_name]
            if feat.dim() == 4 and feat.shape[-1] == expected_ch and feat.shape[1] != expected_ch:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            result[stage_name] = feat
        return result

    @property
    def output_shape(self) -> Dict[str, Tuple[int, int, int]]:
        return {
            name: (
                self._stage_out_channels[name],
                self._stage_out_strides[name],
                self._stage_out_strides[name],
            )
            for name in self.out_features
        }
