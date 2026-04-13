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
        num_classes = int(config.model.magformer.sem_seg_head.num_classes)
        if num_classes != 1:
            raise ValueError(
                "This project currently supports exactly one foreground class. Multi-class is not implemented. "
                f"Set model.magformer.sem_seg_head.num_classes=1 (got {num_classes})."
            )

        from .magformer import MagFormerArch

        # Pass the full config so MagFormerArch can resolve both:
        # - new nested keys (model.magformer.*)
        # - legacy flat keys normalized by MagFormerConfig before validation
        return MagFormerArch.from_config(config)
    elif arch in _EXTERNAL_BASELINE_ARCHS:
        raise ValueError(
            "External baselines are not built through magformer.models.build. "
            "Use the dedicated wrappers under `baselines/` or `scripts/experiments/`."
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
