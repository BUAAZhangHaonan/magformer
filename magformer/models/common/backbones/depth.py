# -*- coding: utf-8 -*-
"""
Lightweight depth backbone helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


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


class TimmDepthBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        out_features: List[str],
        pretrained: bool = False,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        import timm

        self.model_name = str(model_name)
        self.out_features = list(out_features)

        probe = timm.create_model(
            self.model_name,
            pretrained=pretrained and (weights_path is None),
            features_only=True,
            in_chans=1,
        )
        feature_info = probe.feature_info
        all_reductions = [int(x) for x in feature_info.reduction()]
        all_channels = [int(x) for x in feature_info.channels()]
        out_indices = select_timm_out_indices(all_reductions, self.out_features)
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
        state_dict = torch.load(weights_path, map_location="cpu")
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
        self.model.load_state_dict(filtered_state, strict=False)

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


def build_depth_backbone(
    *,
    depth_mode: str,
    backbone_cfg: Any,
    convnext_cfg: Optional[Any] = None,
    default_out_features: Optional[List[str]] = None,
) -> nn.Module:
    from .convnext import ConvNeXtDepth

    backbone_name = str(getattr(backbone_cfg, "name", "ConvNeXtDepthBackbone"))
    out_features = list(getattr(backbone_cfg, "out_features", None) or default_out_features or ["res3"])
    pretrained = bool(getattr(backbone_cfg, "pretrained", False))
    weights_path = getattr(backbone_cfg, "weights", None)

    if depth_mode == "legacy":
        if backbone_name not in {"ConvNeXtDepthBackbone", "convnext", "convnext_depth", "legacy_convnext"}:
            raise ValueError(
                f"Legacy depth_mode only supports ConvNeXtDepthBackbone, got: {backbone_name}"
            )
        if convnext_cfg is None:
            raise ValueError("convnext_cfg is required for legacy depth backbone")
        return ConvNeXtDepth(
            depths=convnext_cfg.depths,
            dims=convnext_cfg.dims,
            drop_path_rate=convnext_cfg.drop_path_rate,
            layer_scale=convnext_cfg.layer_scale,
            out_features=out_features,
            pretrained=pretrained,
            weights_path=weights_path,
        )

    timm_name_map = {
        "mobilenetv3smalldepthbackbone": "mobilenetv3_small_100",
        "mobilenetv3_small": "mobilenetv3_small_100",
        "mobilenetv3_small_100": "mobilenetv3_small_100",
        "mobilenetv3largedepthbackbone": "mobilenetv3_large_100",
        "mobilenetv3_large": "mobilenetv3_large_100",
        "mobilenetv3_large_100": "mobilenetv3_large_100",
        "resnet18depthbackbone": "resnet18",
        "resnet18": "resnet18",
        "convnextlitedepthbackbone": "convnext_tiny",
        "convnext_tiny": "convnext_tiny",
        "convnext_small": "convnext_small",
    }
    name_key = backbone_name.replace("_", "").lower()
    if name_key not in timm_name_map and backbone_name.lower() not in timm_name_map:
        raise ValueError(f"Unsupported light depth backbone: {backbone_name}")

    return TimmDepthBackbone(
        model_name=timm_name_map.get(name_key, timm_name_map[backbone_name.lower()]),
        out_features=out_features,
        pretrained=pretrained,
        weights_path=weights_path,
    )
