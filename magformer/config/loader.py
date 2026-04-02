# -*- coding: utf-8 -*-
"""
MAGFormer Configuration Loader

YAML 配置文件加载器，支持命令行参数覆盖。
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml
import torch

from .schema import (
    MagFormerConfig,
    merge_configs,
)


# =============================================================================
# 默认配置
# =============================================================================
DEFAULT_CONFIG_PATHS = [
    "configs/base.yaml",
    "configs/magformer.yaml",
]


# =============================================================================
# 配置加载
# =============================================================================
def load_yaml_file(path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件。

    Args:
        path: YAML 文件路径

    Returns:
        配置字典
    """
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def save_yaml_file(config: Dict[str, Any], path: str) -> None:
    """
    保存配置到 YAML 文件。

    Args:
        config: 配置字典
        path: 保存路径
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def load_config(
    config_file: str,
    cli_args: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> MagFormerConfig:
    """
    加载配置文件并解析为 MagFormerConfig。

    Args:
        config_file: 配置文件路径
        cli_args: 命令行参数列表 (sys.argv[1:])
        overrides: 额外的覆盖配置字典

    Returns:
        MagFormerConfig 对象
    """
    # 加载 YAML 配置
    config_dict = load_yaml_file(config_file)

    # 解析命令行覆盖
    if cli_args is not None:
        cli_overrides = parse_cli_overrides(cli_args)
        config_dict = merge_configs(config_dict, cli_overrides)

    # 应用额外覆盖
    if overrides is not None:
        config_dict = merge_configs(config_dict, overrides)

    # 转换为 Pydantic 模型
    config = MagFormerConfig(**config_dict)

    return config


def parse_cli_overrides(args: List[str]) -> Dict[str, Any]:
    """
    解析命令行参数覆盖。

    支持格式:
        --key value
        --key.nested value
        --key=value
        --key.nested=value

    Args:
        args: 命令行参数列表

    Returns:
        覆盖配置字典
    """
    overrides = {}
    i = 0

    while i < len(args):
        arg = args[i]

        if arg.startswith("--"):
            # 移除 -- 前缀
            key_value = arg[2:]

            # 处理 --key=value 格式
            if "=" in key_value:
                key, value = key_value.split("=", 1)
                overrides = _set_nested_value(
                    overrides, key, _parse_value(value))
                i += 1
                continue

            # 处理 --key value 格式
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                overrides = _set_nested_value(
                    overrides, key_value, _parse_value(value))
                i += 2
                continue

            # 处理布尔标志 --flag
            overrides = _set_nested_value(overrides, key_value, True)
            i += 1
        else:
            i += 1

    return overrides


def _parse_value(value: str) -> Any:
    """
    解析值字符串为正确的 Python 类型。

    Args:
        value: 值字符串

    Returns:
        解析后的值
    """
    # 布尔值
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False

    # 整数
    try:
        return int(value)
    except ValueError:
        pass

    # 浮点数
    try:
        return float(value)
    except ValueError:
        pass

    # 列表 (逗号分隔)
    if "," in value:
        return [_parse_value(v.strip()) for v in value.split(",")]

    # 字符串
    return value


def _set_nested_value(config: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    """
    设置嵌套字典值。

    Args:
        config: 配置字典
        key: 点分隔的键 (如 "model.rgb_backbone.name")
        value: 值

    Returns:
        更新后的配置字典
    """
    keys = key.split(".")
    current = config

    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    current[keys[-1]] = value
    return config


# =============================================================================
# 命令行参数解析
# =============================================================================
def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="MAGFormer Training and Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 必需参数
    parser.add_argument(
        "--config",
        "--config-file",
        dest="config",
        type=str,
        required=True,
        help="Path to config file",
    )

    # 数据覆盖
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="Dataset root directory (overrides data.dataset_root)",
    )

    # 模型覆盖
    parser.add_argument(
        "--weights",
        type=str,
        help="Path to model weights (overrides model.weights)",
    )
    parser.add_argument(
        "--finetune-weights",
        type=str,
        help="Path to warm-start model weights (overrides model.finetune_weights)",
    )

    # 训练覆盖
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides runtime.output_dir)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint (overrides runtime.resume)",
    )

    # 运行模式
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only",
    )

    # GPU 设置
    parser.add_argument(
        "--gpus",
        type=str,
        help="GPU IDs to use (e.g., '0,1,2')",
    )

    # 其他
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of data loading workers",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )

    return parser.parse_known_args()[0]


# =============================================================================
# 工具函数
# =============================================================================
def setup_device(runtime_config: "RuntimeConfig") -> torch.device:
    """
    设置计算设备。

    Args:
        runtime_config: 运行时配置

    Returns:
        PyTorch 设备对象
    """
    if runtime_config.device == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        return torch.device("cpu")

    gpu_id = runtime_config.gpus[0] if runtime_config.gpus else 0
    return torch.device(f"cuda:{gpu_id}")


def setup_logging(runtime_config: "RuntimeConfig") -> None:
    """
    设置日志系统。

    Args:
        runtime_config: 运行时配置
    """
    import logging

    log_dir = Path(runtime_config.logger.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 配置根日志器
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件处理器
    file_handler = logging.FileHandler(log_dir / "train.log")
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def set_seed(seed: int) -> None:
    """
    设置随机种子以保证可复现性。

    Args:
        seed: 随机种子
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保卷积操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
