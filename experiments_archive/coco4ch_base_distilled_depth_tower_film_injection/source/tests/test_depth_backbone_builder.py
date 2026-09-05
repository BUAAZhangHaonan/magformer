from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("timm")

from magformer.models.common.backbones.convnext import ConvNeXtDepth
from magformer.models.common.backbones.depth import build_depth_backbone
from magformer.models.common.backbones.mobilenet import MobileNetV3Depth
from magformer.models.common.backbones.resnet import ResNetDepth


def _backbone_cfg(name: str, out_features: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        out_features=list(out_features or ["res3"]),
        pretrained=False,
        weights=None,
    )


def _convnext_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale=1e-6,
    )


def test_build_depth_backbone_returns_specific_backbone_classes() -> None:
    convnext = build_depth_backbone(
        depth_mode="legacy",
        backbone_cfg=_backbone_cfg("ConvNeXtDepthBackbone", ["res2", "res3", "res4", "res5"]),
        convnext_cfg=_convnext_cfg(),
        default_out_features=["res3"],
    )
    mobilenet = build_depth_backbone(
        depth_mode="light",
        backbone_cfg=_backbone_cfg("MobileNetV3SmallDepthBackbone"),
        convnext_cfg=_convnext_cfg(),
        default_out_features=["res3"],
    )
    resnet = build_depth_backbone(
        depth_mode="light",
        backbone_cfg=_backbone_cfg("ResNet18DepthBackbone"),
        convnext_cfg=_convnext_cfg(),
        default_out_features=["res3"],
    )

    assert isinstance(convnext, ConvNeXtDepth)
    assert isinstance(mobilenet, MobileNetV3Depth)
    assert isinstance(resnet, ResNetDepth)


def test_legacy_mode_is_compatibility_flag_not_convnext_lock() -> None:
    model = build_depth_backbone(
        depth_mode="legacy",
        backbone_cfg=_backbone_cfg("ResNet18DepthBackbone"),
        convnext_cfg=_convnext_cfg(),
        default_out_features=["res3"],
    )
    assert isinstance(model, ResNetDepth)
