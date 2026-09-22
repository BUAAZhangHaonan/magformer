# -*- coding: utf-8 -*-
"""
MAGFormer Model Library

纯 PyTorch 实现的模型库，包含 MAGFormer 及其组件。
"""

from .build import build_model

# 可用的模型
__all__ = [
    "build_model",
]
