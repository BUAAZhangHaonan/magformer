# -*- coding: utf-8 -*-
"""
MAGFormer Data Loading

纯 PyTorch 数据加载模块，支持 COCO 格式的 RGB-D 实例分割数据集。
"""

from .dataset import CocoRgbdDataset
from .transforms import RGBDTransform, Compose
from .collate import collate_fn
from .semi_supervised_dataset import SemiSupervisedDataset

__all__ = [
    "CocoRgbdDataset",
    "RGBDTransform",
    "Compose",
    "collate_fn",
    "SemiSupervisedDataset",
]
