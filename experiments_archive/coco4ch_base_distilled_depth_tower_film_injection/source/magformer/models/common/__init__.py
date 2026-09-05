# -*- coding: utf-8 -*-
"""
MAGFormer Common Components

通用模型组件，包括 backbone、层和 transformer 组件。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SwinTransformer",
    "D2SwinBackbone",
    "ResidualDepthSwinBackbone",
    "LightDepthPyramid",
    "ConvNeXtDepth",
    "MobileNetV3Depth",
    "ResNetDepth",
    "TimmDepthBackbone",
    "build_depth_backbone",
    "select_timm_out_indices",
    "SimpleTransformerDecoder",
    "SimplePixelDecoder",
    "MultiScaleMaskedTransformerDecoder",
    "MSDeformAttnPixelDecoder",
    "HungarianMatcher",
    "SetCriterion",
    "RGBDPointRefiner",
]


def __getattr__(name: str) -> Any:
    if name == "SwinTransformer":
        from .backbones.swin import SwinTransformer

        return SwinTransformer
    if name == "D2SwinBackbone":
        from .backbones.d2_swin import D2SwinBackbone

        return D2SwinBackbone
    if name in {"ResidualDepthSwinBackbone", "LightDepthPyramid"}:
        from .backbones.residual_depth_swin import (
            LightDepthPyramid,
            ResidualDepthSwinBackbone,
        )

        return {
            "ResidualDepthSwinBackbone": ResidualDepthSwinBackbone,
            "LightDepthPyramid": LightDepthPyramid,
        }[name]
    if name == "ConvNeXtDepth":
        from .backbones.convnext import ConvNeXtDepth

        return ConvNeXtDepth
    if name == "MobileNetV3Depth":
        from .backbones.mobilenet import MobileNetV3Depth

        return MobileNetV3Depth
    if name == "ResNetDepth":
        from .backbones.resnet import ResNetDepth

        return ResNetDepth
    if name in {"TimmDepthBackbone", "build_depth_backbone", "select_timm_out_indices"}:
        from .backbones.depth import (
            TimmDepthBackbone,
            build_depth_backbone,
        )
        from .backbones.depth_base import select_timm_out_indices

        return {
            "TimmDepthBackbone": TimmDepthBackbone,
            "build_depth_backbone": build_depth_backbone,
            "select_timm_out_indices": select_timm_out_indices,
        }[name]
    if name == "SimpleTransformerDecoder":
        from .transformer.decoder import SimpleTransformerDecoder

        return SimpleTransformerDecoder
    if name == "SimplePixelDecoder":
        from .pixel_decoder import SimplePixelDecoder

        return SimplePixelDecoder
    if name == "MultiScaleMaskedTransformerDecoder":
        from .transformer.multiscale_decoder import MultiScaleMaskedTransformerDecoder

        return MultiScaleMaskedTransformerDecoder
    if name == "MSDeformAttnPixelDecoder":
        from .pixel_decoder_msdeformattn import MSDeformAttnPixelDecoder

        return MSDeformAttnPixelDecoder
    if name == "HungarianMatcher":
        from .matcher import HungarianMatcher

        return HungarianMatcher
    if name == "SetCriterion":
        from .criterion import SetCriterion

        return SetCriterion
    if name == "RGBDPointRefiner":
        from .mask_refiner import RGBDPointRefiner

        return RGBDPointRefiner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
