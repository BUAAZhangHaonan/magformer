# -*- coding: utf-8 -*-
"""
MAGFormer Model

纯 PyTorch 实现的 MAGFormer 模型，包括模态融合、解码器等组件。
"""

from .fusion import ModalityFusionModule
from .arch import MagFormerArch

__all__ = [
    "ModalityFusionModule",
    "MagFormerArch",
]
