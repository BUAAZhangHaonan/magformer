# -*- coding: utf-8 -*-
"""
Training Utilities

日志、度量计算和其他训练辅助工具。
支持 TensorBoard 和 WandB。
"""

import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import torch
import torch.nn as nn
import numpy as np


logger = logging.getLogger(__name__)
def load_torch_checkpoint(
    filename: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load a tensor-only PyTorch checkpoint."""
    filepath = Path(filename).expanduser().resolve()
    return torch.load(filepath, map_location=map_location, weights_only=True)


# =============================================================================
# 日志设置
# =============================================================================
def setup_logger(
    name: str = "magformer",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    设置日志记录器。

    Args:
        name: 日志器名称
        log_file: 日志文件路径
        level: 日志级别

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除现有处理器
    logger.handlers.clear()

    # 格式化器
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TensorBoardLogger:
    """TensorBoard 日志记录器"""

    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.available = True
        except ImportError:
            print("Warning: tensorboard not available, logging disabled")
            self.available = False
            self.writer = None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """记录标量"""
        if self.available and self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """记录多个标量"""
        if self.available and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        """记录图像"""
        if self.available and self.writer is not None:
            self.writer.add_image(tag, image, step)

    def log_images(self, tag: str, images: torch.Tensor, step: int, dataformats: str = "NCHW") -> None:
        """记录图像网格"""
        if self.available and self.writer is not None:
            self.writer.add_images(tag, images, step, dataformats=dataformats)

    def close(self) -> None:
        """关闭日志记录器"""
        if self.available and self.writer is not None:
            self.writer.close()


class WandBLogger:
    """Weights & Biases 日志记录器"""

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        dir: Optional[str] = None,
    ):
        """
        Args:
            project: 项目名称
            entity: 实体名称
            name: 运行名称
            config: 配置字典
            dir: 输出目录
        """
        self.available = False
        self.run = None

        try:
            import wandb

            self.available = True
            self.wandb = wandb

            # 初始化运行
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                dir=dir,
            )

            print(f"[WandB] Initialized run: {self.run.name}")
        except ImportError:
            print("Warning: wandb not available, logging disabled")

    def log_scalar(self, key: str, value: float, step: int) -> None:
        """记录标量"""
        if self.available and self.run is not None:
            self.wandb.log({key: value}, step=step)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """记录多个标量"""
        if self.available and self.run is not None:
            self.wandb.log(metrics, step=step)

    def log_image(self, key: str, image: np.ndarray, step: int) -> None:
        """记录图像"""
        if self.available and self.run is not None:
            self.wandb.log({key: self.wandb.Image(image)}, step=step)

    def log_images(self, key: str, images: List[np.ndarray], step: int) -> None:
        """记录图像列表"""
        if self.available and self.run is not None:
            self.wandb.log({key: [self.wandb.Image(img)
                           for img in images]}, step=step)

    def finish(self) -> None:
        """结束运行"""
        if self.available and self.run is not None:
            self.wandb.finish()


class CombinedLogger:
    """组合日志记录器 (TensorBoard + WandB)"""

    def __init__(
        self,
        tensorboard_dir: str,
        wandb_project: str,
        wandb_entity: Optional[str] = None,
        wandb_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = True,
    ):
        """
        Args:
            tensorboard_dir: TensorBoard 日志目录
            wandb_project: WandB 项目名称
            wandb_entity: WandB 实体名称
            wandb_name: WandB 运行名称
            config: 配置字典
            use_tensorboard: 是否使用 TensorBoard
            use_wandb: 是否使用 WandB
        """
        self.tb_logger = None
        self.wandb_logger = None

        if use_tensorboard:
            self.tb_logger = TensorBoardLogger(tensorboard_dir)

        if use_wandb:
            self.wandb_logger = WandBLogger(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                config=config,
            )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """记录标量"""
        if self.tb_logger is not None:
            self.tb_logger.log_scalar(tag, value, step)
        if self.wandb_logger is not None:
            self.wandb_logger.log_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """记录多个标量"""
        if self.tb_logger is not None:
            self.tb_logger.log_scalars(main_tag, tag_scalar_dict, step)
        if self.wandb_logger is not None:
            # WandB 不需要 main_tag，直接记录所有指标
            self.wandb_logger.log_scalars(tag_scalar_dict, step)

    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int) -> None:
        """记录图像"""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if self.tb_logger is not None:
            self.tb_logger.log_image(tag, image, step)
        if self.wandb_logger is not None:
            self.wandb_logger.log_image(tag, image.numpy(), step)

    def close(self) -> None:
        """关闭所有日志记录器"""
        if self.tb_logger is not None:
            self.tb_logger.close()
        if self.wandb_logger is not None:
            self.wandb_logger.finish()


# =============================================================================
# 度量工具
# =============================================================================
class AverageMeter:
    """计算和存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """重置所有统计"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        更新统计。

        Args:
            val: 新值
            n: 样本数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-8)


class ProgressMeter:
    """显示训练进度"""

    def __init__(self, num_batches: int, meters: Dict[str, AverageMeter], prefix: str = ""):
        """
        Args:
            num_batches: 总批次数
            meters: 度量字典 {名称: AverageMeter}
            prefix: 显示前缀
        """
        self.batch_fmtstr = "{:d} [{:" + str(len(str(num_batches))) + \
            "d}/{:" + str(len(str(num_batches))) + "d}]"
        self.meters = meters
        self.num_batches = num_batches
        self.prefix = prefix

    def display(self, batch: int) -> str:
        """
        获取进度显示字符串。

        Args:
            batch: 当前批次

        Returns:
            格式化的进度字符串
        """
        entries = [self.prefix +
                   self.batch_fmtstr.format(batch, self.num_batches)]
        entries += [str(meter) for meter in self.meters.values()]
        print("\t".join(entries))
        return "\t".join(entries)

    def display_summary(self) -> str:
        """显示总结"""
        entries = [" * " + self.prefix]
        entries += [str(meter) for meter in self.meters.values()]
        print(" ".join(entries))
        return " ".join(entries)


# =============================================================================
# 模型工具
# =============================================================================
def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    计算模型参数数量。

    Args:
        model: PyTorch 模型
        trainable_only: 是否只统计可训练参数

    Returns:
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    获取优化器当前学习率。

    Args:
        optimizer: PyTorch 优化器

    Returns:
        当前学习率
    """
    if len(optimizer.param_groups) == 0:
        return 0.0
    # Return the largest LR among groups so logs reflect the main (non-backbone) group.
    return max(float(param_group.get("lr", 0.0)) for param_group in optimizer.param_groups)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    设置随机种子。

    Args:
        seed: 随机种子
        deterministic: 是否启用确定性模式 (较慢)
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def save_checkpoint(
    state: Dict[str, Any],
    filename: str,
) -> None:
    """
    保存检查点。

    Args:
        state: 状态字典
        filename: 文件名
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, filepath)



def _assert_checkpoint_values_equal(
    actual: Any,
    expected: Any,
    *,
    actual_name: str,
    expected_name: str,
) -> None:
    """Require exact recursive identity for two checkpoint state values."""
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        if not (torch.is_tensor(actual) and torch.is_tensor(expected)):
            raise RuntimeError(
                f"Best checkpoint state mismatch: {actual_name} and "
                f"{expected_name} have different value types"
            )
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise RuntimeError(
                f"Best checkpoint state mismatch: {actual_name} and "
                f"{expected_name} have different tensor metadata"
            )
        if not torch.equal(actual.cpu(), expected.cpu()):
            raise RuntimeError(
                f"Best checkpoint state mismatch: {actual_name} does not "
                f"exactly equal {expected_name}"
            )
        return

    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not (isinstance(actual, Mapping) and isinstance(expected, Mapping)):
            raise RuntimeError(
                f"Best checkpoint state mismatch: {actual_name} and "
                f"{expected_name} have different value types"
            )
        if set(actual) != set(expected):
            missing = sorted(set(expected).difference(actual), key=str)
            extra = sorted(set(actual).difference(expected), key=str)
            raise RuntimeError(
                f"Best checkpoint state mismatch: {actual_name} and "
                f"{expected_name} have different keys; missing={missing}, "
                f"extra={extra}"
            )
        for key in actual:
            _assert_checkpoint_values_equal(
                actual[key],
                expected[key],
                actual_name=f"{actual_name}.{key}",
                expected_name=f"{expected_name}.{key}",
            )
        return

    if type(actual) is not type(expected) or actual != expected:
        raise RuntimeError(
            f"Best checkpoint state mismatch: {actual_name} does not exactly "
            f"equal {expected_name}"
        )


def _assert_checkpoint_state_compatible(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    actual_name: str,
    expected_name: str,
) -> None:
    """Require state dictionaries to describe the same parameter structure."""
    if set(actual) != set(expected):
        missing = sorted(set(expected).difference(actual), key=str)
        extra = sorted(set(actual).difference(expected), key=str)
        raise RuntimeError(
            f"Best checkpoint state mismatch: {actual_name} and "
            f"{expected_name} have different keys; missing={missing}, extra={extra}"
        )
    for key in actual:
        actual_value = actual[key]
        expected_value = expected[key]
        if not (torch.is_tensor(actual_value) and torch.is_tensor(expected_value)):
            raise RuntimeError(
                f"Best checkpoint state mismatch: {actual_name}.{key} and "
                f"{expected_name}.{key} must both be tensors"
            )
        if (
            actual_value.dtype != expected_value.dtype
            or actual_value.shape != expected_value.shape
        ):
            raise RuntimeError(
                f"Best checkpoint state mismatch: {actual_name}.{key} and "
                f"{expected_name}.{key} have different tensor metadata"
            )


def validate_best_model_artifact(checkpoint: Any) -> None:
    """Validate that a best artifact's primary weights match its provenance."""
    if not isinstance(checkpoint, Mapping):
        return
    if checkpoint.get("artifact_kind") != "model_best":
        return

    version = checkpoint.get("artifact_format_version")
    if version != 2:
        raise RuntimeError(
            "Best checkpoint artifact_format_version must equal 2; legacy or "
            "missing provenance metadata is ambiguous"
        )

    source = checkpoint.get("evaluated_weight_source")
    if source not in {"raw", "ema"}:
        raise RuntimeError(
            "Best checkpoint evaluated_weight_source must be exactly 'raw' or "
            f"'ema', got {source!r}"
        )

    primary = checkpoint.get("model_state_dict")
    raw_state = checkpoint.get("raw_model_state_dict")
    if not isinstance(primary, Mapping):
        raise RuntimeError("Best checkpoint model_state_dict must be a mapping")
    if not isinstance(raw_state, Mapping):
        raise RuntimeError("Best checkpoint raw_model_state_dict must be a mapping")

    ema_state = checkpoint.get("ema_state_dict")
    ema_shadow = None
    if ema_state is not None:
        if not isinstance(ema_state, Mapping):
            raise RuntimeError("Best checkpoint ema_state_dict must be a mapping")
        ema_shadow = ema_state.get("shadow")
        if not isinstance(ema_shadow, Mapping):
            raise RuntimeError(
                "Best checkpoint ema_state_dict.shadow must be a mapping"
            )

    _assert_checkpoint_state_compatible(
        raw_state,
        primary,
        actual_name="raw_model_state_dict",
        expected_name="model_state_dict",
    )

    if source == "ema":
        if ema_shadow is None:
            raise RuntimeError(
                "Best checkpoint claims evaluated_weight_source='ema' but has "
                "no valid ema_state_dict.shadow"
            )
        _assert_checkpoint_values_equal(
            primary,
            ema_shadow,
            actual_name="model_state_dict",
            expected_name="ema_state_dict.shadow",
        )
    else:
        _assert_checkpoint_values_equal(
            primary,
            raw_state,
            actual_name="model_state_dict",
            expected_name="raw_model_state_dict",
        )


def load_checkpoint(
    filename: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    加载检查点。

    Args:
        filename: 检查点文件路径
        model: 模型
        optimizer: 优化器 (可选)
        strict: 是否严格加载权重

    Returns:
        检查点字典
    """
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = load_torch_checkpoint(filepath, map_location="cpu")
    validate_best_model_artifact(checkpoint)

    def _strip_module_prefix_if_needed(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(state_dict, dict):
            return state_dict
        if not state_dict:
            return state_dict
        if not all(isinstance(k, str) for k in state_dict.keys()):
            return state_dict
        if not any(k.startswith("module.") for k in state_dict.keys()):
            return state_dict
        return {
            k[7:] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }

    # 加载模型权重
    _incompat = None
    if "model_state_dict" in checkpoint:
        _incompat = model.load_state_dict(
            _strip_module_prefix_if_needed(checkpoint["model_state_dict"]),
            strict=strict,
        )
    elif "state_dict" in checkpoint:
        _incompat = model.load_state_dict(
            _strip_module_prefix_if_needed(checkpoint["state_dict"]),
            strict=strict,
        )
    else:
        _incompat = model.load_state_dict(
            _strip_module_prefix_if_needed(checkpoint),
            strict=strict,
        )

    # 加载优化器状态
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"[Checkpoint] Loaded checkpoint from {filepath}")
    return checkpoint, _incompat


def clip_gradients(model: nn.Module, max_norm: float, norm_type: float = 2.0) -> float:
    """
    裁剪模型梯度。

    Args:
        model: PyTorch 模型
        max_norm: 最大范数
        norm_type: 范数类型 (1=L1, 2=L2)

    Returns:
        裁剪前的梯度范数
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0

    if norm_type == float("inf"):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0.0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    return total_norm
