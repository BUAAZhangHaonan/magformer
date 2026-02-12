# -*- coding: utf-8 -*-
"""
MAGFormer Configuration System

纯 PyTorch 配置系统，使用 YAML + Pydantic 管理配置。
"""

from .schema import (
    DataConfig,
    ModelConfig,
    SolverConfig,
    RuntimeConfig,
    MagFormerConfig,
    BaseModelConfig,
)
from .loader import load_config, merge_configs, parse_args, setup_device, set_seed

__all__ = [
    "load_config",
    "merge_configs",
    "parse_args",
    "setup_device",
    "set_seed",
    "DataConfig",
    "ModelConfig",
    "SolverConfig",
    "RuntimeConfig",
    "MagFormerConfig",
    "BaseModelConfig",
]
