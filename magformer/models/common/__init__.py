# -*- coding: utf-8 -*-
"""
MAGFormer Common Components

通用模型组件，包括 backbone、层和 transformer 组件。
"""

# Backbones
from .backbones.swin import SwinTransformer
from .backbones.convnext import ConvNeXtDepth

# Layers
from .transformer.decoder import SimpleTransformerDecoder
from .pixel_decoder import SimplePixelDecoder
from .matcher import HungarianMatcher
from .criterion import SetCriterion

__all__ = [
    "SwinTransformer",
    "ConvNeXtDepth",
    "SimpleTransformerDecoder",
    "SimplePixelDecoder",
    "HungarianMatcher",
    "SetCriterion",
]
