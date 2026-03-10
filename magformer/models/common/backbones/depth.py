# -*- coding: utf-8 -*-
"""
Depth backbone registry and compatibility builder.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch.nn as nn

from .convnext import ConvNeXtDepth
from .depth_base import TimmDepthBackboneBase, select_timm_out_indices
from .mobilenet import MobileNetV3Depth
from .resnet import ResNetDepth


TimmDepthBackbone = TimmDepthBackboneBase


def _normalized_name(name: str) -> str:
    return str(name).replace("-", "_").replace(" ", "_").lower()
def build_depth_backbone(
    *,
    depth_mode: str,
    backbone_cfg: Any,
    convnext_cfg: Optional[Any] = None,
    default_out_features: Optional[List[str]] = None,
) -> nn.Module:
    """
    Build a depth backbone from config.

    `depth_mode` is a compatibility semantic flag:
    - `legacy`: preserve old MagFormer heavy-depth defaults and config assumptions
    - `light`: use the lightweight experiment line defaults

    The concrete backbone implementation is selected by `depth_backbone.name`,
    not by `depth_mode`.
    """

    del depth_mode

    backbone_name = str(getattr(backbone_cfg, "name", "ConvNeXtDepthBackbone"))
    out_features = list(
        getattr(backbone_cfg, "out_features", None) or default_out_features or ["res3"]
    )
    pretrained = bool(getattr(backbone_cfg, "pretrained", False))
    weights_path = getattr(backbone_cfg, "weights", None)
    name_key = _normalized_name(backbone_name)

    if name_key in {"convnextdepthbackbone", "convnext", "convnext_depth", "legacy_convnext"}:
        if convnext_cfg is None:
            raise ValueError("convnext_cfg is required for ConvNeXt depth backbone")
        return ConvNeXtDepth(
            depths=convnext_cfg.depths,
            dims=convnext_cfg.dims,
            drop_path_rate=convnext_cfg.drop_path_rate,
            layer_scale=convnext_cfg.layer_scale,
            out_features=out_features,
            pretrained=pretrained,
            weights_path=weights_path,
        )

    if name_key in {
        "mobilenetv3smalldepthbackbone",
        "mobilenetv3_small",
        "mobilenetv3_small_100",
        "mobilenetv3largedepthbackbone",
        "mobilenetv3_large",
        "mobilenetv3_large_100",
    }:
        variant = "large" if "large" in name_key else "small"
        return MobileNetV3Depth(
            variant=variant,
            out_features=out_features,
            pretrained=pretrained,
            weights_path=weights_path,
        )

    if name_key in {"resnet18depthbackbone", "resnet18"}:
        return ResNetDepth(
            depth=18,
            out_features=out_features,
            pretrained=pretrained,
            weights_path=weights_path,
        )

    raise ValueError(f"Unsupported depth backbone: {backbone_name}")


def build_depth_backbone_from_name(
    *,
    backbone_name: str,
    out_features: List[str],
    pretrained: bool = False,
    weights_path: Optional[str] = None,
    convnext_cfg: Optional[Any] = None,
) -> nn.Module:
    cfg = type(
        "DepthBackboneCfg",
        (),
        {
            "name": backbone_name,
            "out_features": list(out_features),
            "pretrained": bool(pretrained),
            "weights": weights_path,
        },
    )()
    return build_depth_backbone(
        depth_mode="light",
        backbone_cfg=cfg,
        convnext_cfg=convnext_cfg,
        default_out_features=out_features,
    )
