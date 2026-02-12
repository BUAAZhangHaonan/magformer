# -*- coding: utf-8 -*-
"""
Model Factory

模型构建工厂函数，根据配置创建模型实例。
"""

from typing import Any, Dict
import torch.nn as nn

from ..config import MagFormerConfig


def build_model(config: MagFormerConfig) -> nn.Module:
    """
    根据配置构建模型。

    Args:
        config: MAGFormer 配置对象

    Returns:
        模型实例
    """
    arch = config.model.meta_architecture

    if arch == "MagFormer":
        from .magformer import MagFormerArch

        return MagFormerArch.from_config(config.model.magformer)
    elif arch in ["ucn", "UCN"]:
        from .baselines.ucn import UCNModel

        return UCNModel(config.model.baseline)
    elif arch in ["msmformer", "MSMFormer"]:
        from .baselines.msmformer import MSMFormerModel

        return MSMFormerModel(config.model.baseline)
    elif arch in ["uoa_is", "UOAIS", "uoa-is"]:
        from .baselines.uoa_is import UOAISModel

        return UOAISModel(config.model.baseline)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
