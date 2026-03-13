# -*- coding: utf-8 -*-
"""
Model Factory

模型构建工厂函数，根据配置创建模型实例。
"""

import torch.nn as nn

from ..config import MagFormerConfig


_EXTERNAL_BASELINE_ARCHS = {
    "ucn",
    "UCN",
    "msmformer",
    "MSMFormer",
    "uoa_is",
    "UOAIS",
    "uoa-is",
}


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

        # Pass the full config so MagFormerArch can resolve both:
        # - new nested keys (model.magformer.*)
        # - legacy root-level compatibility keys (e.g. dpe_enabled/dpe_beta)
        return MagFormerArch.from_config(config)
    elif arch in _EXTERNAL_BASELINE_ARCHS:
        raise ValueError(
            "External baselines are not built through magformer.models.build. "
            "Use the dedicated wrappers under `baselines/` or `scripts/experiments/`."
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
