# -*- coding: utf-8 -*-
"""
MAGFormer Training Engine

纯 PyTorch 实现的通用训练器。
支持 AMP、DDP、Checkpoint 管理和 TensorBoard/WandB 日志。
"""

import time
import json
import csv
import math
import copy
import random
from importlib import metadata as importlib_metadata
from collections.abc import Mapping
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .utils import (
    AverageMeter,
    save_checkpoint,
    load_torch_checkpoint,
    validate_best_model_artifact,
    get_lr,
    CombinedLogger,
)
from .coco_export import outputs_to_coco_instances
from .model_ema import ModelEMA
from .eval_runtime import run_inference_evaluation

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =============================================================================
# Trainer
# =============================================================================
from contextlib import nullcontext


def _as_cuda_amp(enabled: bool, device: torch.device) -> bool:
    """Return true only when CUDA AMP can actually run."""
    return bool(enabled and device.type == "cuda" and torch.cuda.is_available())


class _CpuTensorSnapshot:
    """CPU tensor clone plus its original device for transactional rollback."""

    __slots__ = ("tensor", "device")

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor.detach().to(device="cpu", copy=True)
        self.device = tensor.device


class Trainer:
    """
    通用训练器。

    特性:
    - AMP 混合精度训练
    - DDP 多 GPU 支持
    - 自动 Checkpoint 管理
    - TensorBoard + WandB 日志
    - 学习率调度
    - 梯度裁剪
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        val_dataset: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        device: torch.device = torch.device("cuda"),
        output_dir: str = "output",
        max_iter: int = 100000,
        eval_period: int = 5000,
        checkpoint_period: int = 5000,
        log_period: int = 100,
        amp_enabled: bool = True,
        clip_gradients: bool = True,
        clip_value: float = 1.0,
        resume: Optional[str] = None,
        logger_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model: 模型
            criterion: 损失函数
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            val_dataset: 验证数据集（用于获取COCO对象计算mAP）
            config: 配置字典
            device: 计算设备
            output_dir: 输出目录
            max_iter: 最大迭代数
            eval_period: 评估周期
            checkpoint_period: 检查点保存周期
            log_period: 日志记录周期
            amp_enabled: 是否启用 AMP
            clip_gradients: 是否裁剪梯度
            clip_value: 梯度裁剪值
            resume: 恢复检查点路径
            logger_config: 日志配置
        """
        self.model = model.to(device)
        if isinstance(criterion, nn.Module):
            self.criterion = criterion.to(device)
        else:
            self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.config = config or {}
        self.device = device
        self.output_dir = Path(output_dir)
        self.max_iter = max_iter
        self.eval_period = eval_period
        self.checkpoint_period = checkpoint_period
        self.log_period = log_period
        self.amp_enabled = _as_cuda_amp(amp_enabled, self.device)
        self.clip_gradients = clip_gradients
        self.clip_value = clip_value
        # Runtime/solver config (needed early for step-unit and accumulation settings)
        runtime_cfg = self.config.get("runtime", {}) if isinstance(self.config, dict) else {}
        solver_cfg = self.config.get("solver", {}) if isinstance(self.config, dict) else {}
        if not isinstance(solver_cfg, dict):
            raise TypeError("solver must be a mapping produced by SolverConfig")
        self.iteration_unit = str(
            solver_cfg.get("iteration_unit", "optimizer_step")
        )
        if self.iteration_unit not in {"optimizer_step", "legacy_micro_step"}:
            raise ValueError(
                "solver.iteration_unit must be 'optimizer_step' or "
                f"'legacy_micro_step', got {self.iteration_unit!r}"
            )
        # Early stopping is explicit and opt-in. `best_metric` remains the
        # checkpoint-selection metric; early stopping owns separate state.
        self.early_stop = False
        self.early_stop_best_metric = float("-inf")
        self._patience_counter = 0
        es_cfg = runtime_cfg.get("early_stop", {})
        if not isinstance(es_cfg, dict):
            raise TypeError(
                "runtime.early_stop must be a mapping produced by EarlyStopConfig"
            )
        if "target_ap" in es_cfg:
            raise ValueError(
                "runtime.early_stop.target_ap must be normalized by the config "
                "loader; use runtime.early_stop.target"
            )
        self.early_stop_enabled = bool(es_cfg.get("enabled", False))
        self._early_stop_monitor = str(es_cfg.get("monitor", "val/mAP")).strip()
        self._patience_limit = int(es_cfg.get("patience", 5))
        self._min_delta = float(es_cfg.get("min_delta", 0.001))
        early_stop_target = es_cfg.get("target", None)
        self._early_stop_target = (
            None if early_stop_target is None else float(early_stop_target)
        )
        self._early_stop_min_optimizer_step = int(
            es_cfg.get("min_optimizer_step", 0)
        )
        if not self._early_stop_monitor:
            raise ValueError("runtime.early_stop.monitor must not be empty")
        if self._patience_limit <= 0:
            raise ValueError("runtime.early_stop.patience must be positive")
        if not math.isfinite(self._min_delta) or self._min_delta < 0.0:
            raise ValueError(
                "runtime.early_stop.min_delta must be finite and non-negative"
            )
        if self._early_stop_target is not None and not math.isfinite(
            self._early_stop_target
        ):
            raise ValueError("runtime.early_stop.target must be finite when set")
        if self._early_stop_min_optimizer_step < 0:
            raise ValueError(
                "runtime.early_stop.min_optimizer_step must be non-negative"
            )
        if (
            self.early_stop_enabled
            and self._early_stop_monitor == "val/mAP"
            and self._early_stop_target is not None
            and not 0.0 <= self._early_stop_target <= 1.0
        ):
            raise ValueError(
                "runtime.early_stop.target must be in [0, 1] when monitor=val/mAP"
            )
        # Eval config: iou_types and max_images for faster eval during training
        self.eval_iou_types = runtime_cfg.get("eval_iou_types", None)
        self.eval_max_images = runtime_cfg.get("eval_max_images", None)
        self.eval_max_dets = int(runtime_cfg.get("eval_max_dets", 100))
        self.eval_amp_enabled = bool(runtime_cfg.get("eval_amp_enabled", False))
        # Gradient accumulation
        self.grad_accum_steps = int(runtime_cfg.get("grad_accum_steps", 1))
        if self.grad_accum_steps <= 0:
            raise ValueError(
                f"runtime.grad_accum_steps must be positive, got {self.grad_accum_steps}"
            )
        amp_init_scale = runtime_cfg.get("amp_init_scale", 65536.0)
        if isinstance(amp_init_scale, bool) or not isinstance(
            amp_init_scale, (int, float)
        ):
            raise ValueError("runtime.amp_init_scale must be a real number")
        self.amp_init_scale = float(amp_init_scale)
        if not math.isfinite(self.amp_init_scale) or self.amp_init_scale <= 0.0:
            raise ValueError(
                "runtime.amp_init_scale must be finite and positive, got "
                f"{self.amp_init_scale!r}"
            )
        max_consecutive_amp_skips = runtime_cfg.get(
            "max_consecutive_amp_skips", 16
        )
        if (
            type(max_consecutive_amp_skips) is not int
            or max_consecutive_amp_skips < 0
        ):
            raise ValueError(
                "runtime.max_consecutive_amp_skips must be a non-negative int"
            )
        self.max_consecutive_amp_skips = max_consecutive_amp_skips
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")
        if (
            self.iteration_unit == "legacy_micro_step"
            and self.max_iter % self.grad_accum_steps != 0
        ):
            raise ValueError(
                f"max_iter={self.max_iter} must be divisible by "
                f"grad_accum_steps={self.grad_accum_steps}; training completion "
                "must occur on an optimizer-step boundary"
            )
        for period_name, period_value in (
            ("eval_period", self.eval_period),
            ("checkpoint_period", self.checkpoint_period),
            ("log_period", self.log_period),
        ):
            if period_value <= 0:
                raise ValueError(f"{period_name} must be positive, got {period_value}")
            if (
                self.iteration_unit == "legacy_micro_step"
                and period_name != "log_period"
                and period_value <= self.max_iter
                and period_value % self.grad_accum_steps != 0
            ):
                raise ValueError(
                    f"{period_name}={period_value} must be divisible by "
                    f"grad_accum_steps={self.grad_accum_steps} when it can trigger "
                    "during training"
                )
        self._accum_count = 0
        if self.grad_accum_steps > 1:
            print(f"[Trainer] Gradient accumulation: {self.grad_accum_steps} steps")
        self.metrics_log_file = self.output_dir / "metrics_log.jsonl"
        self.metrics_csv_file = self.output_dir / "metrics_log.csv"
        self.visualization_dir = self.output_dir / "visualizations"
        self.peak_memory_file = self.output_dir / "peak_memory_mb.txt"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self._csv_header_written = self.metrics_csv_file.exists()

        # 分布式训练
        self.distributed = False
        self.world_size = 1
        self.rank = 0

        # 训练状态
        # ``data_epoch`` belongs to the data pipeline, not the optimization
        # schedule.  Keep ``start_epoch`` as a compatibility alias for older
        # callers, but only ``data_epoch`` is persisted by checkpoint v3.
        self.data_epoch = 0
        self.start_epoch = 0
        self._loader_state_restored = False
        self.start_iter = 0
        # `current_iter` is retained as the legacy/public micro-step counter.
        self.current_iter = 0
        self.optimizer_step = 0
        self.amp_skipped_steps = 0
        self.consecutive_amp_skips = 0
        self._last_preclip_grad_norm: Optional[float] = None
        self._last_grad_finite: Optional[bool] = None
        self._last_amp_scale_before_step: Optional[float] = None
        self._last_amp_scale_after_step: Optional[float] = None
        self._last_nonfinite_grad_param: Optional[str] = None
        self._last_nonfinite_grad_count = 0
        self.best_metric = float("-inf")
        self._train_start_monotonic: Optional[float] = None
        self._iter_time_window_sec = deque(maxlen=20)
        self._pbar = None

        # 设置日志
        self.logger = self._setup_logger(logger_config)

        # AMP Scaler
        self.scaler = (
            GradScaler("cuda", init_scale=self.amp_init_scale)
            if self.amp_enabled
            else None
        )

        # EMA (Exponential Moving Average)
        self.ema = None
        if runtime_cfg.get("ema_enabled", False):
            configured_ema_warmup = int(runtime_cfg.get("ema_warmup_iters", 200))
            if configured_ema_warmup < 0:
                raise ValueError("runtime.ema_warmup_iters must be non-negative")
            if self.iteration_unit == "legacy_micro_step":
                if configured_ema_warmup % self.grad_accum_steps != 0:
                    raise ValueError(
                        "runtime.ema_warmup_iters must be divisible by "
                        f"grad_accum_steps={self.grad_accum_steps}, got "
                        f"{configured_ema_warmup}"
                    )
                ema_warmup_optimizer_steps = (
                    configured_ema_warmup // self.grad_accum_steps
                )
            else:
                ema_warmup_optimizer_steps = configured_ema_warmup
            self.ema = ModelEMA(
                self.model,
                decay=float(runtime_cfg.get("ema_decay", 0.9999)),
                warmup_iters=ema_warmup_optimizer_steps,
            )
            print(
                f"[Trainer] EMA enabled: decay={runtime_cfg.get('ema_decay', 0.9999)}, "
                f"iteration_unit={self.iteration_unit}, "
                f"configured_warmup={configured_ema_warmup}, "
                f"warmup_optimizer_steps={ema_warmup_optimizer_steps}"
            )

        # 恢复训练
        if resume is not None:
            self.resume(resume)

    def _iteration_step(self) -> int:
        if self.iteration_unit == "optimizer_step":
            return int(self.optimizer_step)
        return int(self.current_iter)

    def _event_due(self, period: int, *, successful_optimizer_step: bool) -> bool:
        if self.iteration_unit == "optimizer_step":
            return (
                successful_optimizer_step
                and self.optimizer_step > 0
                and self.optimizer_step % period == 0
            )
        return self.current_iter > 0 and self.current_iter % period == 0

    def _setup_logger(self, logger_config: Optional[Dict[str, Any]]) -> CombinedLogger:
        """设置日志记录器"""
        if logger_config is None:
            logger_config = {}

        log_type = logger_config.get("type", "tensorboard")
        tb_dir = str(self.output_dir / "logs" / "tensorboard")
        wandb_project = logger_config.get("project", "magformer")
        wandb_entity = logger_config.get("entity", None)
        wandb_name = logger_config.get("run_name", None)

        # 确定使用哪些日志器
        use_tb = log_type in ["tensorboard", "both"]
        use_wandb = log_type in ["wandb", "both"]
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            use_tb = False
            use_wandb = False

        return CombinedLogger(
            tensorboard_dir=tb_dir,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_name=wandb_name,
            config=self.config,
            use_tensorboard=use_tb,
            use_wandb=use_wandb,
        )

    def _early_stop_policy_state(self) -> Dict[str, Any]:
        return {
            "enabled": self.early_stop_enabled,
            "monitor": self._early_stop_monitor,
            "patience": self._patience_limit,
            "min_delta": self._min_delta,
            "target": self._early_stop_target,
            "min_optimizer_step": self._early_stop_min_optimizer_step,
        }

    @staticmethod
    def _validate_checkpoint_metric(value: Any, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(
                f"Checkpoint {field_name} must be a real number, got "
                f"{type(value).__name__}"
            )
        metric = float(value)
        if math.isfinite(metric) or metric == float("-inf"):
            return metric
        raise RuntimeError(
            f"Checkpoint {field_name} must be finite or the initial -inf, "
            f"got {metric}"
        )

    @staticmethod
    def _require_checkpoint_int(
        checkpoint: Mapping[str, Any], field_name: str
    ) -> int:
        if field_name not in checkpoint:
            raise RuntimeError(f"Checkpoint format v3 is missing {field_name}")
        value = checkpoint[field_name]
        if type(value) is not int:
            raise RuntimeError(
                f"Checkpoint {field_name} must have type int, got "
                f"{type(value).__name__}"
            )
        return value

    @staticmethod
    def _require_checkpoint_mapping(
        checkpoint: Mapping[str, Any], field_name: str
    ) -> Mapping[str, Any]:
        if field_name not in checkpoint:
            raise RuntimeError(f"Checkpoint format v3 is missing {field_name}")
        value = checkpoint[field_name]
        if not isinstance(value, Mapping):
            raise RuntimeError(
                f"Checkpoint {field_name} must be a mapping, got "
                f"{type(value).__name__}"
            )
        return value

    @staticmethod
    def _normalize_resume_model_state(
        state_dict: Mapping[str, Any]
    ) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in state_dict.items():
            if not isinstance(key, str):
                raise RuntimeError("Checkpoint model_state_dict keys must be strings")
            normalized_key = key[7:] if key.startswith("module.") else key
            if normalized_key in normalized:
                raise RuntimeError(
                    "Checkpoint model_state_dict contains colliding normalized key: "
                    f"{normalized_key}"
                )
            normalized[normalized_key] = value
        return normalized

    def _early_stop_state_dict(self) -> Dict[str, Any]:
        return {
            **self._early_stop_policy_state(),
            "early_stop_best_metric": self.early_stop_best_metric,
            "patience_counter": self._patience_counter,
            "should_stop": self.early_stop,
        }

    def _validate_early_stop_state(
        self, checkpoint: Mapping[str, Any]
    ) -> Dict[str, Any]:
        state = checkpoint.get("early_stop_state")
        if state is None:
            if self.early_stop_enabled:
                raise RuntimeError(
                    "Checkpoint does not contain early_stop_state and cannot be "
                    "fully resumed with runtime.early_stop.enabled=true"
                )
            return {
                "early_stop_best_metric": float("-inf"),
                "patience_counter": 0,
                "should_stop": False,
            }
        if not isinstance(state, Mapping):
            raise RuntimeError("Checkpoint early_stop_state must be a mapping")

        expected_policy = self._early_stop_policy_state()
        missing_policy = [key for key in expected_policy if key not in state]
        if missing_policy:
            raise RuntimeError(
                "Checkpoint early_stop_state is missing policy fields: "
                + ", ".join(missing_policy)
            )
        for key, expected_value in expected_policy.items():
            if state[key] != expected_value:
                raise RuntimeError(
                    "Checkpoint early-stop policy does not match the current config: "
                    f"{key} checkpoint={state[key]!r}, config={expected_value!r}"
                )

        dynamic_fields = {
            "early_stop_best_metric",
            "patience_counter",
            "should_stop",
        }
        missing_dynamic = sorted(dynamic_fields.difference(state))
        if missing_dynamic:
            raise RuntimeError(
                "Checkpoint early_stop_state is missing dynamic fields: "
                + ", ".join(missing_dynamic)
            )

        early_stop_best_metric = self._validate_checkpoint_metric(
            state["early_stop_best_metric"],
            "early_stop_state.early_stop_best_metric",
        )
        patience_counter = state["patience_counter"]
        if type(patience_counter) is not int or patience_counter < 0:
            raise RuntimeError(
                "Checkpoint early-stop patience_counter must be a non-negative integer"
            )
        should_stop = state["should_stop"]
        if type(should_stop) is not bool:
            raise RuntimeError("Checkpoint early-stop should_stop must be a boolean")
        if should_stop and not self.early_stop_enabled:
            raise RuntimeError(
                "Checkpoint cannot have should_stop=true while early stopping is disabled"
            )
        if not self.early_stop_enabled and (
            early_stop_best_metric != float("-inf") or patience_counter != 0
        ):
            raise RuntimeError(
                "Checkpoint disabled early-stop state must keep the initial "
                "best=-inf and patience_counter=0"
            )
        if early_stop_best_metric == float("-inf") and (
            patience_counter != 0 or should_stop
        ):
            raise RuntimeError(
                "Checkpoint initial early-stop best=-inf requires "
                "patience_counter=0 and should_stop=false"
            )
        if patience_counter > self._patience_limit:
            raise RuntimeError(
                "Checkpoint early-stop patience_counter cannot exceed patience"
            )
        if not should_stop and patience_counter >= self._patience_limit:
            raise RuntimeError(
                "Checkpoint non-terminal early-stop state must have "
                "patience_counter < patience"
            )
        if (
            should_stop
            and self._early_stop_target is None
            and patience_counter != self._patience_limit
        ):
            raise RuntimeError(
                "Checkpoint target-free stopped state must stop exactly at patience"
            )

        return {
            "early_stop_best_metric": early_stop_best_metric,
            "patience_counter": patience_counter,
            "should_stop": should_stop,
        }

    def _apply_early_stop_state(self, state: Mapping[str, Any]) -> None:
        self.early_stop_best_metric = state["early_stop_best_metric"]
        self._patience_counter = state["patience_counter"]
        self.early_stop = state["should_stop"]

    def _restore_early_stop_state(self, checkpoint: Mapping[str, Any]) -> None:
        self._apply_early_stop_state(self._validate_early_stop_state(checkpoint))

    def _update_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Update rank-0 early-stop state from one eligible evaluation."""
        if not self.early_stop_enabled or self.early_stop:
            return self.early_stop
        if self.optimizer_step < self._early_stop_min_optimizer_step:
            return False
        if self._early_stop_monitor not in metrics:
            raise KeyError(
                "Early-stop monitor is missing from evaluation metrics: "
                f"{self._early_stop_monitor}"
            )

        current_metric = float(metrics[self._early_stop_monitor])
        if not math.isfinite(current_metric):
            raise ValueError(
                f"Early-stop monitor {self._early_stop_monitor} must be finite, "
                f"got {current_metric}"
            )

        previous_best = self.early_stop_best_metric
        improved = current_metric > previous_best + self._min_delta
        reached_target = (
            self._early_stop_target is not None
            and current_metric >= self._early_stop_target
        )

        if improved:
            self.early_stop_best_metric = current_metric
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        if reached_target:
            self.early_stop = True
            self._console_log(
                f"[EARLY STOP] {self._early_stop_monitor}={current_metric:.4f} "
                f">= target {self._early_stop_target:.4f}"
            )
        elif not improved and self._patience_counter >= self._patience_limit:
            self.early_stop = True
            self._console_log(
                f"[EARLY STOP] {self._early_stop_monitor} plateaued for "
                f"{self._patience_limit} eligible evals "
                f"(best={previous_best:.4f}, current={current_metric:.4f})"
            )
        return self.early_stop

    def _sync_early_stop_state(self, rank_zero_error: Optional[str] = None) -> None:
        """Broadcast rank-0 early-stop state or a finalization error to all workers."""
        if not self.distributed:
            return
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "Distributed trainer cannot synchronize early stopping without "
                "an initialized process group"
            )
        payload = [None]
        if self.rank == 0:
            payload[0] = {
                "error": rank_zero_error,
                "state": self._early_stop_state_dict(),
            }
        dist.broadcast_object_list(
            payload,
            src=0,
            device=self.device,
        )
        received = payload[0]
        if not isinstance(received, dict):
            raise RuntimeError("Invalid early-stop synchronization payload")
        if set(received) != {"error", "state"}:
            raise RuntimeError(
                "Early-stop synchronization payload must contain error and state"
            )
        self._restore_early_stop_state({"early_stop_state": received["state"]})
        error = received["error"]
        if error is not None:
            if not isinstance(error, str):
                raise RuntimeError("Early-stop synchronization error must be a string")
            raise RuntimeError(f"Rank-0 evaluation finalization failed: {error}")

    def train(self) -> None:
        """主训练循环"""
        self.model.train()

        for method_name in ("is_epoch_exhausted", "start_next_epoch"):
            if not callable(getattr(self.train_loader, method_name, None)):
                raise RuntimeError(
                    "Exact training requires ExactStatefulDataLoader."
                    f"{method_name}()"
                )

        if self._train_start_monotonic is None:
            self._train_start_monotonic = time.monotonic()
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

        # A restored StatefulDataLoader already owns the sampler epoch and
        # cursor.  Calling set_epoch here would discard that exact position.
        sampler = getattr(self.train_loader, "sampler", None)
        if not self._loader_state_restored and callable(
            getattr(sampler, "set_epoch", None)
        ):
            sampler.set_epoch(self.data_epoch)

        data_iter = iter(self.train_loader)
        self._loader_state_restored = False

        self._console_log(
            f"[{self._now_console_ts()}] start "
            f"{self.iteration_unit}={self._iteration_step()}/{self.max_iter} "
            f"micro_step={self.current_iter} optimizer_step={self.optimizer_step} "
            f"eval_period={self.eval_period} ckpt_period={self.checkpoint_period} "
            f"log_period={self.log_period}"
        )

        pbar = None
        if tqdm is not None:
            pbar = tqdm(total=self.max_iter, initial=self._iteration_step(),
                        desc="MAGFormer Train", dynamic_ncols=True)
        self._pbar = pbar

        while self._iteration_step() < self.max_iter and not self.early_stop:
            # 获取下一个批次
            try:
                batch = next(data_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "Training loader reached StopIteration before MAGFormer "
                    "canonicalized its exhausted epoch"
                ) from exc

            # 训练一个批次
            optimizer_step_before = self.optimizer_step
            losses = self._train_step(batch)
            successful_optimizer_step = self.optimizer_step > optimizer_step_before

            # Canonicalize an epoch boundary before evaluation or checkpointing.
            # A boundary checkpoint therefore always describes the next epoch
            # at cursor 0 and never a stale iterator waiting for StopIteration.
            if self.train_loader.is_epoch_exhausted():
                next_data_epoch = self.data_epoch + 1
                data_iter = self.train_loader.start_next_epoch(next_data_epoch)
                self.data_epoch = next_data_epoch
                self.start_epoch = next_data_epoch

            if pbar is not None:
                if self.iteration_unit == "legacy_micro_step":
                    pbar.update(1)
                elif successful_optimizer_step:
                    pbar.update(1)
                pbar.set_postfix({
                    "micro": self.current_iter,
                    "opt": self.optimizer_step,
                    "loss": f"{losses['total_loss'].item():.4f}",
                    "lr": f"{get_lr(self.optimizer):.6f}",
                })

            eval_due = self._event_due(
                self.eval_period,
                successful_optimizer_step=successful_optimizer_step,
            )
            checkpoint_due = self._event_due(
                self.checkpoint_period,
                successful_optimizer_step=successful_optimizer_step,
            )

            # Persist the milestone state before evaluation or visualization.
            # Evaluation errors must not discard the resumable checkpoint.
            if eval_due or checkpoint_due:
                self.save_checkpoint()

            # 评估
            if eval_due:
                self.evaluate()
                if self.early_stop:
                    self._console_log(
                        f"[EARLY STOP] Stopping at "
                        f"{self.iteration_unit}={self._iteration_step()}"
                    )
                    break

        # 训练结束
        if pbar is not None:
            pbar.close()
        self._pbar = None
        self._console_log(f"[{self._now_console_ts()}] training completed")
        # Final evaluation if not already evaluated at this iteration
        if self._iteration_step() % self.eval_period != 0:
            self._console_log(
                f"[{self._now_console_ts()}] running final evaluation at "
                f"{self.iteration_unit}={self._iteration_step()}"
            )
            eval_result = self.evaluate()
            # BUG 8 fix: evaluate() already calls _finalize_eval_result()
            # internally; do NOT call it again on the returned dict.
        peak_memory_mb = self._current_peak_memory_mb()
        if peak_memory_mb is not None:
            self.peak_memory_file.write_text(
                f"{peak_memory_mb:.2f}\n", encoding="utf-8")
        # Best weights are managed during evaluation. The final training state
        # overwrites the same resumable checkpoint used by periodic saves.
        self.save_checkpoint()
        self.logger.close()

    @staticmethod
    def _gradient_values(gradient: torch.Tensor) -> torch.Tensor:
        if gradient.is_sparse:
            return gradient.detach().coalesce().values()
        return gradient.detach()

    def _preclip_gradient_norm(self) -> float:
        gradients = []
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            gradient = self._gradient_values(parameter.grad).float()
            if gradient.numel() == 0:
                continue
            gradients.append(gradient)
        if not gradients:
            return 0.0

        # Scale before squaring so a mathematically finite L2 norm cannot
        # overflow merely because the gradients are stored in FP32.  Only the
        # scalar reconstruction uses FP64; gradient tensors are not duplicated
        # in FP64 and the method performs one host synchronization.
        max_abs = torch.stack([gradient.abs().max() for gradient in gradients]).max()
        safe_scale = torch.where(
            torch.isfinite(max_abs) & (max_abs > 0),
            max_abs,
            torch.ones_like(max_abs),
        )
        scaled_squares = torch.stack(
            [((gradient / safe_scale) ** 2).sum() for gradient in gradients]
        ).sum()
        total_norm = max_abs.double() * scaled_squares.double().sqrt()
        return float(total_norm.item())

    def _scan_nonfinite_gradients(self) -> tuple[Optional[str], int]:
        first_name = None
        nonfinite_count = 0
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            values = self._gradient_values(parameter.grad)
            count = int((~torch.isfinite(values)).sum().item())
            if count > 0 and first_name is None:
                first_name = name
            nonfinite_count += count
        return first_name, nonfinite_count

    def _clip_gradients_with_known_norm(self, grad_norm: float) -> None:
        clip_coefficient = self.clip_value / (grad_norm + 1.0e-6)
        if clip_coefficient >= 1.0:
            return
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(clip_coefficient)

    @staticmethod
    def _validate_finite_losses(losses: Mapping[str, Any]) -> None:
        for name, value in losses.items():
            if not torch.is_tensor(value):
                continue
            nonfinite_count = int((~torch.isfinite(value.detach())).sum().item())
            if nonfinite_count > 0:
                raise FloatingPointError(
                    f"Loss {name} contains {nonfinite_count} non-finite value(s)"
                )

    def _step_optimizer(self) -> bool:
        """Run one optimizer attempt and report whether parameters were updated."""
        if self.amp_enabled:
            previous_scale = float(self.scaler.get_scale())
            self._last_amp_scale_before_step = previous_scale
            self.scaler.step(self.optimizer)
            self.scaler.update()
            current_scale = float(self.scaler.get_scale())
            self._last_amp_scale_after_step = current_scale
            optimizer_stepped = current_scale >= previous_scale

            if optimizer_stepped:
                if hasattr(self.optimizer, "_opt_called"):
                    self.optimizer._opt_called = True
                if (
                    hasattr(self.optimizer, "_step_count")
                    and self.optimizer._step_count == 0
                ):
                    self.optimizer._step_count = 1
            return optimizer_stepped

        self._last_amp_scale_before_step = None
        self._last_amp_scale_after_step = None
        self.optimizer.step()
        return True

    def _notify_model_successful_optimizer_step(self) -> None:
        model = getattr(self.model, "module", self.model)
        hook = getattr(model, "on_successful_optimizer_step", None)
        if callable(hook):
            hook()

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        训练一个批次。

        Args:
            batch: 输入批次字典
        """
        iter_start = time.perf_counter()
        amp_limit_error: Optional[FloatingPointError] = None
        successful_optimizer_step = False
        # 数据移到设备
        images = batch["images"].to(self.device, non_blocking=True)
        depths = batch["depths"].to(self.device, non_blocking=True)
        depth_valid_masks = batch.get("depth_valid_masks", None)
        if depth_valid_masks is not None:
            depth_valid_masks = depth_valid_masks.to(self.device, non_blocking=True)
        noise_masks = batch.get("noise_masks", None)
        if noise_masks is not None:
            noise_masks = noise_masks.to(self.device, non_blocking=True)
        padding_masks = batch.get("padding_masks", None)
        if padding_masks is not None:
            padding_masks = padding_masks.to(self.device, non_blocking=True)
        targets = batch.get("targets", None)

        if targets is not None:
            # 处理目标 (根据模型需求)
            targets = self._prepare_targets(targets, batch)

        # 前向传播
        amp_context = autocast("cuda") if self.amp_enabled else nullcontext()
        with amp_context:
            outputs = self.model(
                images,
                depths,
                targets,
                padding_masks=padding_masks,
                depth_valid_masks=depth_valid_masks,
                depth_noise_masks=noise_masks,
            )
            losses = self._compute_losses(outputs, targets)
        self._validate_finite_losses(losses)

        # --- Gradient accumulation ---
        # Zero gradients at the start of each accumulation window
        if self._accum_count == 0:
            self.optimizer.zero_grad()

        # Scale loss for accumulation (gradients are averaged over accum_steps)
        accum_loss = losses["total_loss"] / self.grad_accum_steps

        # Backward pass with DDP no_sync for intermediate accumulation steps
        is_last_accum = (self._accum_count == self.grad_accum_steps - 1)
        use_no_sync = (not is_last_accum) and self.distributed and hasattr(self.model, "no_sync")

        if use_no_sync:
            with self.model.no_sync():
                if self.amp_enabled:
                    self.scaler.scale(accum_loss).backward()
                else:
                    accum_loss.backward()
        else:
            if self.amp_enabled:
                self.scaler.scale(accum_loss).backward()
            else:
                accum_loss.backward()

        self._accum_count += 1

        # Only step optimizer when accumulation is complete
        if self._accum_count >= self.grad_accum_steps:
            # Unscale first so diagnostics and clipping use real gradients.
            if self.amp_enabled:
                self.scaler.unscale_(self.optimizer)
            grad_norm = self._preclip_gradient_norm()
            grad_finite = math.isfinite(grad_norm)
            self._last_grad_finite = grad_finite
            self._last_preclip_grad_norm = grad_norm if grad_finite else None

            if not grad_finite and not self.amp_enabled:
                raise FloatingPointError(
                    f"preclip_grad_norm must be finite, got {grad_norm!r}"
                )
            if grad_finite and self.clip_gradients:
                self._clip_gradients_with_known_norm(grad_norm)

            # 优化器步进
            optimizer_stepped = self._step_optimizer()

            # Reset accumulation counter
            self._accum_count = 0

            # M1+M2 fix: LR scheduler and EMA only advance on real optimizer steps.
            # Gated by optimizer_stepped so AMP skip-on-inf also skips LR/EMA.
            if optimizer_stepped:
                successful_optimizer_step = True
                self.consecutive_amp_skips = 0
                self._last_nonfinite_grad_param = None
                self._last_nonfinite_grad_count = 0
                self._notify_model_successful_optimizer_step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # EMA warmup uses a zero-based optimizer-update index: the first
                # successful optimizer update is index 0 and therefore has zero
                # decay during warmup.
                if self.ema is not None:
                    self.ema.update(self.optimizer_step, self.model)
                self.optimizer_step += 1
            else:
                self.amp_skipped_steps += 1
                self.consecutive_amp_skips += 1
                (
                    self._last_nonfinite_grad_param,
                    self._last_nonfinite_grad_count,
                ) = self._scan_nonfinite_gradients()
                parameter_name = self._last_nonfinite_grad_param or "<unknown>"
                self._console_log(
                    "[AMP overflow] "
                    f"micro_step={self.current_iter + 1} "
                    f"consecutive={self.consecutive_amp_skips}/"
                    f"{self.max_consecutive_amp_skips} "
                    f"first_nonfinite_param={parameter_name} "
                    f"nonfinite_grad_count={self._last_nonfinite_grad_count} "
                    f"grad_finite={grad_finite} "
                    f"scale={self._last_amp_scale_before_step}->"
                    f"{self._last_amp_scale_after_step}"
                )
                if self.consecutive_amp_skips > self.max_consecutive_amp_skips:
                    amp_limit_error = FloatingPointError(
                        "Exceeded runtime.max_consecutive_amp_skips: "
                        f"consecutive={self.consecutive_amp_skips}, "
                        f"limit={self.max_consecutive_amp_skips}, "
                        f"first_nonfinite_param={parameter_name}, "
                        f"nonfinite_grad_count={self._last_nonfinite_grad_count}, "
                        f"scale_before={self._last_amp_scale_before_step}, "
                        f"scale_after={self._last_amp_scale_after_step}"
                    )

        iter_time_sec = time.perf_counter() - iter_start
        self._iter_time_window_sec.append(iter_time_sec)

        self.current_iter += 1

        if amp_limit_error is not None:
            raise amp_limit_error

        if self._event_due(
            self.log_period,
            successful_optimizer_step=successful_optimizer_step,
        ):
            self._log_training(losses)

        return losses

    def _append_metrics_log(self, metrics: Dict[str, float], phase: str) -> None:
        if self._train_start_monotonic is None:
            # Ensure logs always have coherent elapsed time even if called outside `train()`.
            self._train_start_monotonic = time.monotonic()

        now_wall = time.time()
        now_mono = time.monotonic()
        elapsed_sec = now_mono - self._train_start_monotonic

        iter_time_sec = None
        eta_sec = None
        if len(self._iter_time_window_sec) > 0:
            iter_time_sec = sum(self._iter_time_window_sec) / \
                len(self._iter_time_window_sec)
            iter_done = self._iteration_step()
            remaining = max(0, self.max_iter - int(iter_done))
            eta_sec = iter_time_sec * remaining

        runtime_telemetry = self._current_runtime_telemetry()
        payload = {
            "iter": self._iteration_step(),
            "iteration_unit": self.iteration_unit,
            "micro_step": int(self.current_iter),
            "optimizer_step": int(self.optimizer_step),
            "amp_skipped_steps": int(self.amp_skipped_steps),
            "consecutive_amp_skips": int(self.consecutive_amp_skips),
            "phase": phase,
            "wall_time": float(now_wall),
            "wall_time_iso": self._now_iso(),
            "elapsed_sec": float(elapsed_sec),
            "iter_time_sec": None if iter_time_sec is None else float(iter_time_sec),
            "eta_sec": None if eta_sec is None else float(eta_sec),
            "peak_memory_mb": self._current_peak_memory_mb(),
            **{k: float(v) for k, v in metrics.items()},
            **runtime_telemetry,
        }

        with open(self.metrics_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        with open(self.metrics_csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(payload)

    def _finalize_eval_result(self, result: Any) -> Dict[str, float]:
        """Apply the shared post-inference evaluation finalization path.

        The same helper is used for single-GPU evaluation and DDP rank 0 after
        predictions have been gathered, so metric logging and best-checkpoint
        selection stay equivalent.
        """
        log_dict = result.log_dict
        coco_metrics = result.coco_metrics
        canonical_segm_ap = None
        if "val/segm_AP" in log_dict:
            raw_segm_ap = log_dict["val/segm_AP"]
            if isinstance(raw_segm_ap, bool):
                raise ValueError("Validation metric val/segm_AP must be a finite number")
            try:
                canonical_segm_ap = float(raw_segm_ap)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Validation metric val/segm_AP must be a finite number"
                ) from exc
            if not math.isfinite(canonical_segm_ap):
                raise ValueError(
                    "Validation metric val/segm_AP must be finite, got "
                    f"{canonical_segm_ap}"
                )

        if result.visualization_batch is not None and result.visualization_outputs is not None:
            self._save_eval_visualization(result.visualization_batch, result.visualization_outputs)

        if log_dict:
            self.logger.log_scalars("val", log_dict, self.optimizer_step)
            self._append_metrics_log(log_dict, phase="val")

        summary = "no_metrics"
        if "val/segm_AP" in log_dict:
            summary = f"segm_AP={log_dict['val/segm_AP']:.4f}"
        elif "val/mAP" in log_dict:
            summary = f"mAP={log_dict['val/mAP']:.4f}"
        self._console_log(f"[{self._now_console_ts()}] eval_summary {summary}")

        if coco_metrics:
            self._console_log(self._format_coco_metrics(coco_metrics))
        if result.coco_results_path is not None:
            self._console_log(
                f"[{self._now_console_ts()}] coco_results {result.coco_results_path}"
            )
        if "val/diag_score_p50" in log_dict:
            self._console_log(
                f"[{self._now_console_ts()}] eval_diag "
                f"score_p50={log_dict.get('val/diag_score_p50', 0.0):.4f} "
                f"score_p90={log_dict.get('val/diag_score_p90', 0.0):.4f} "
                f"bbox_area_ratio_p50={log_dict.get('val/diag_bbox_area_ratio_p50', 0.0):.4f} "
                f"bbox_area_ratio_p90={log_dict.get('val/diag_bbox_area_ratio_p90', 0.0):.4f} "
                f"mask_nonempty_ratio={log_dict.get('val/diag_mask_nonempty_ratio', 0.0):.4f}"
            )

        # Rank 0 updates early-stop state before any best-checkpoint write, so
        # the checkpoint contains the decision made from the previous state.
        self._update_early_stopping(log_dict)

        if canonical_segm_ap is not None:
            if canonical_segm_ap > self.best_metric:
                self.best_metric = canonical_segm_ap

        return log_dict

    def _prepare_targets(self, targets: List[Dict[str, torch.Tensor]], batch: Dict[str, torch.Tensor]) -> Any:
        """
        准备目标数据以供模型使用。

        Args:
            targets: COCO 格式目标列表
            batch: 输入批次

        Returns:
            模型格式的目标
        """
        # 将每个目标字典中的 tensor 移到设备上
        prepared = []
        for tgt in targets:
            new_tgt = {}
            for k, v in tgt.items():
                if torch.is_tensor(v):
                    new_tgt[k] = v.to(self.device)
                else:
                    new_tgt[k] = v
            prepared.append(new_tgt)
        return prepared

    def _compute_losses(self, outputs: Any, targets: Any) -> Dict[str, torch.Tensor]:
        """
        计算损失。

        Args:
            outputs: 模型输出
            targets: 目标数据

        Returns:
            损失字典
        """
        if self.criterion is None:
            if not isinstance(outputs, dict):
                raise ValueError(
                    "Model outputs must be a loss dict when criterion is None")
            if "total_loss" not in outputs:
                total = None
                for value in outputs.values():
                    if torch.is_tensor(value):
                        total = value if total is None else total + value
                if total is None:
                    raise ValueError("Loss dict contains no tensor losses")
                outputs["total_loss"] = total
            losses = outputs
        elif isinstance(self.criterion, nn.Module):
            losses = {"total_loss": self.criterion(outputs, targets)}
        else:
            losses = self.criterion(outputs, targets)

        return losses

    def _log_training(self, losses: Dict[str, torch.Tensor]) -> None:
        """记录训练日志"""
        lr = get_lr(self.optimizer)
        log_dict = {
            "train/loss": losses["total_loss"].item(),
            "train/lr": lr,
        }

        # 添加其他损失
        for key, value in losses.items():
            if key != "total_loss":
                log_dict[f"train/{key}"] = value.item()

        self.logger.log_scalars("train", log_dict, self.optimizer_step)
        self._append_metrics_log(log_dict, phase="train")

        # Detectron2-like one-line progress
        iter_time_sec = None
        eta_sec = None
        if len(self._iter_time_window_sec) > 0:
            iter_time_sec = sum(self._iter_time_window_sec) / \
                len(self._iter_time_window_sec)
            remaining = max(0, self.max_iter - self._iteration_step())
            eta_sec = iter_time_sec * remaining

        parts = [
            f"[{self._now_console_ts()}]",
            f"{self.iteration_unit}={self._iteration_step()}/{self.max_iter}",
            f"micro_step={self.current_iter}",
            f"optimizer_step={self.optimizer_step}",
            f"eta={self._format_hms(eta_sec)}",
            f"time={self._format_sec(iter_time_sec)}",
            f"lr={lr:.6g}",
            f"loss={losses['total_loss'].item():.4f}",
        ]

        # Prefer aggregated losses (exclude deep supervision terms like *_0, *_1, ...)
        def _is_deep_sup_key(k: str) -> bool:
            head, sep, tail = k.rpartition("_")
            return bool(sep) and tail.isdigit()

        extra_keys = []
        for k in sorted(losses.keys()):
            if k == "total_loss":
                continue
            if not k.startswith("loss_"):
                continue
            if _is_deep_sup_key(k):
                continue
            extra_keys.append(k)

        for k in extra_keys[:6]:
            parts.append(f"{k}={losses[k].item():.4f}")

        self._console_log("  ".join(parts))

    def _save_eval_visualization(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, Any]) -> None:
        """保存 YOLOv8 风格的可视化结果"""
        predictions = outputs.get("predictions", None)
        if predictions is None or len(predictions) == 0:
            return

        import numpy as np
        from ..utils import resolve_class_names
        from ..utils.visualization import visualize_predictions

        images = batch["images"]
        if torch.is_tensor(images):
            img = images[0].detach().cpu().permute(1, 2, 0).numpy()
        else:
            return

        if img.max() <= 1.5:
            img = (img * 255.0).clip(0, 255)
        img = img.astype(np.uint8)

        pred = predictions[0]
        masks = pred.get("masks", [])
        scores = pred.get("scores", [])
        labels = pred.get("category_ids", None)

        if len(masks) == 0:
            import cv2
            save_path = self.visualization_dir / \
                f"eval_iter_{self._iteration_step():07d}_yolov8_empty.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return

        # 转换标签格式
        if labels is not None:
            if hasattr(labels, 'cpu'):
                labels = labels.cpu().tolist()
            elif hasattr(labels, 'numpy'):
                labels = labels.numpy().tolist()

        # 转换掩码格式
        if torch.is_tensor(masks):
            masks = masks.detach().cpu().numpy()
        if isinstance(masks, np.ndarray) and masks.ndim == 3:
            masks = [masks[i] for i in range(masks.shape[0])]
        elif not isinstance(masks, list):
            masks = list(masks)

        # 确保分数是列表
        if hasattr(scores, 'cpu'):
            scores = scores.cpu().tolist()
        elif not isinstance(scores, list):
            scores = list(scores)

        # 使用 YOLOv8 风格可视化
        save_path = str(self.visualization_dir /
                        f"eval_iter_{self._iteration_step():07d}_yolov8.png")
        visualize_predictions(
            image=img,
            masks=masks,
            scores=scores,
            labels=labels,
            class_names=resolve_class_names(config=self.config, dataset=self.val_dataset),
            score_threshold=0.5,
            alpha=0.3,
            show_labels=False,
            show_contours=True,
            contour_thickness=1,
            show_masks=True,
            output_path=save_path,
        )
        print(f"[Trainer] Eval visualization saved: {save_path}")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        评估模型。

        Returns:
            评估指标字典
        """
        if self.val_loader is None:
            return {}

        self._console_log(
            f"[{self._now_console_ts()}] eval "
            f"{self.iteration_unit}={self._iteration_step()}"
        )

        ema = getattr(self, "ema", None)
        evaluated_weight_source = "ema" if ema is not None else "raw"
        evaluated_model_state = None
        raw_model_state = None
        if ema is not None:
            ema.apply_shadow(self.model)

        metrics = {}
        save_best_artifact = False
        try:
            self.model.eval()
            # Verified on 2026-04-13: no supervised loss is computed during validation.
            # Validation is inference-only by design.
            category_ids = list(getattr(self.val_dataset, "category_ids", [])) or None
            result = run_inference_evaluation(
                self.model,
                self.val_loader,
                coco_gt=getattr(self.val_dataset, "coco", None),
                device=self.device,
                optimizer_step=self.optimizer_step,
                output_dir=self.output_dir,
                amp_enabled=self.eval_amp_enabled,
                category_ids=category_ids,
                iou_types=getattr(self, "eval_iou_types", ["bbox", "segm"]),
                max_images=getattr(self, "eval_max_images", None),
                max_dets=getattr(self, "eval_max_dets", 100),
            )
            previous_best_metric = self.best_metric
            metrics = self._finalize_eval_result(result)
            save_best_artifact = self.best_metric > previous_best_metric
            if save_best_artifact:
                evaluated_model_state = self._clone_checkpoint_value_to_cpu(
                    self._model_state_target().state_dict()
                )
                raw_model_state = self._clone_checkpoint_value_to_cpu(
                    ema.backup if ema is not None else evaluated_model_state
                )
        finally:
            if ema is not None:
                ema.restore(self.model)
            self.model.train()

        if save_best_artifact:
            self.save_best_model_artifact(
                metrics,
                evaluated_weight_source=evaluated_weight_source,
                evaluated_model_state_dict=evaluated_model_state,
                raw_model_state_dict=raw_model_state,
            )
        return metrics

    def _convert_to_coco_format(
        self,
        outputs: Dict[str, Any],
        image_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        将模型输出转换为 COCO 评估格式。

        Args:
            outputs: 模型输出字典
            image_ids: 图像 ID 列表

        Returns:
            COCO 格式的预测结果列表
        """
        return outputs_to_coco_instances(
            outputs=outputs,
            image_ids=image_ids,
            score_threshold=0.0,
            mask_threshold=0.5,
            category_offset=1,
        )

    @staticmethod
    def _qualified_name(value: Any) -> str:
        target = getattr(value, "__func__", value)
        module = getattr(target, "__module__", None)
        qualname = getattr(target, "__qualname__", None)
        if isinstance(module, str) and isinstance(qualname, str):
            return f"{module}.{qualname}"
        value_type = type(value)
        return f"{value_type.__module__}.{value_type.__qualname__}"

    @classmethod
    def _canonical_json_value(cls, value: Any, *, path: str) -> Any:
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise TypeError(f"{path} contains a non-finite float")
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            converted = float(value)
            if not math.isfinite(converted):
                raise TypeError(f"{path} contains a non-finite NumPy float")
            return converted
        if isinstance(value, Mapping):
            result = {}
            for key in sorted(value):
                if not isinstance(key, str):
                    raise TypeError(
                        f"{path} contains a non-string mapping key: {key!r}"
                    )
                result[key] = cls._canonical_json_value(
                    value[key], path=f"{path}.{key}"
                )
            return result
        if isinstance(value, (list, tuple)):
            return [
                cls._canonical_json_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        raise TypeError(
            f"{path} contains unsupported type {type(value).__name__}; "
            "training state must have a deterministic checkpoint identity"
        )

    @staticmethod
    def _config_mapping(config: Mapping[str, Any], field: str) -> Dict[str, Any]:
        value = config.get(field, {})
        if not isinstance(value, Mapping):
            raise TypeError(f"Training config {field} must be a mapping")
        return copy.deepcopy(dict(value))

    def _solver_contract_fields(self, field_names: tuple[str, ...]) -> Dict[str, Any]:
        solver = self._config_mapping(self.config, "solver")
        return {
            field: copy.deepcopy(solver[field])
            for field in field_names
            if field in solver
        }

    def _optimizer_contract(self) -> Dict[str, Any]:
        optimizer_solver_fields = (
            "optimizer",
            "base_lr",
            "weight_decay",
            "weight_decay_norm",
            "weight_decay_embed",
            "backbone_multiplier",
            "rgb_backbone_multiplier",
            "depth_backbone_multiplier",
            "mgm_multiplier",
        )
        param_groups = []
        for group in self.optimizer.param_groups:
            static_group = {
                key: copy.deepcopy(value)
                for key, value in group.items()
                if key not in {"params", "lr"}
            }
            param_groups.append(static_group)
        config = {
            "defaults": copy.deepcopy(dict(self.optimizer.defaults)),
            "param_groups": param_groups,
            "solver": self._solver_contract_fields(optimizer_solver_fields),
        }
        return {
            "enabled": True,
            "class": self._qualified_name(self.optimizer),
            "config": self._canonical_json_value(
                config, path="components.optimizer.config"
            ),
        }

    def _scheduler_contract(self) -> Dict[str, Any]:
        enabled = self.lr_scheduler is not None
        scheduler_solver_fields = (
            "lr_scheduler",
            "max_iter",
            "warmup_factor",
            "warmup_iters",
            "warmup_method",
            "steps",
            "gamma",
        )
        if not enabled:
            return {"enabled": False, "class": None, "config": None}
        config = {
            "solver": self._solver_contract_fields(scheduler_solver_fields),
            "grad_accum_steps": self.grad_accum_steps,
        }
        return {
            "enabled": True,
            "class": self._qualified_name(self.lr_scheduler),
            "config": self._canonical_json_value(
                config, path="components.scheduler.config"
            ),
        }

    def _amp_scaler_contract(self) -> Dict[str, Any]:
        enabled = bool(self.amp_enabled)
        scaler_exists = self.scaler is not None
        if scaler_exists != enabled:
            raise RuntimeError(
                "AMP/scaler invariant is broken: "
                f"amp_enabled={enabled}, scaler_exists={scaler_exists}"
            )
        return {
            "enabled": enabled,
            "class": None if not enabled else self._qualified_name(self.scaler),
            "config": {
                "init_scale": self.amp_init_scale,
                "max_consecutive_skips": self.max_consecutive_amp_skips,
            },
        }

    def _ema_contract(self) -> Dict[str, Any]:
        if self.ema is None:
            return {
                "enabled": False,
                "class": None,
                "decay": None,
                "warmup_iters": None,
            }
        decay = getattr(self.ema, "decay", None)
        warmup_iters = getattr(self.ema, "warmup_iters", None)
        if isinstance(decay, bool) or not isinstance(decay, (int, float)):
            raise RuntimeError("EMA decay policy must be a real number")
        decay = float(decay)
        if not math.isfinite(decay):
            raise RuntimeError("EMA decay policy must be finite")
        if type(warmup_iters) is not int or warmup_iters < 0:
            raise RuntimeError("EMA warmup_iters policy must be a non-negative int")
        return {
            "enabled": True,
            "class": self._qualified_name(self.ema),
            "decay": decay,
            "warmup_iters": warmup_iters,
        }

    def _component_contract(self) -> Dict[str, Any]:
        return {
            "optimizer": self._optimizer_contract(),
            "scheduler": self._scheduler_contract(),
            "amp_scaler": self._amp_scaler_contract(),
            "ema": self._ema_contract(),
        }

    def _sampler_contract(self) -> Dict[str, Any]:
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is None:
            raise RuntimeError("Exact resume requires train_loader.sampler")
        sampler_seed = getattr(sampler, "seed", None)
        if type(sampler_seed) is not int:
            generator = getattr(sampler, "generator", None)
            if not isinstance(generator, torch.Generator):
                raise RuntimeError(
                    "Exact resume requires the training sampler to expose an exact "
                    "integer seed or torch.Generator"
                )
            sampler_seed = int(generator.initial_seed())
        return {
            "type": self._qualified_name(sampler),
            "seed": sampler_seed,
        }

    def _build_resume_contract(self) -> Dict[str, Any]:
        loader = self.train_loader
        if loader is None:
            raise RuntimeError("Exact resume requires a training data loader")
        for method_name in (
            "state_dict",
            "load_state_dict",
            "validate_resume_state",
            "is_epoch_exhausted",
            "start_next_epoch",
        ):
            if not callable(getattr(loader, method_name, None)):
                raise RuntimeError(
                    "Exact resume requires a stateful training data loader with "
                    f"{method_name}()"
                )

        batch_size = getattr(loader, "batch_size", None)
        num_workers = getattr(loader, "num_workers", None)
        prefetch_factor = getattr(loader, "prefetch_factor", None)
        persistent_workers = getattr(loader, "persistent_workers", None)
        drop_last = getattr(loader, "drop_last", None)
        if type(batch_size) is not int or batch_size <= 0:
            raise RuntimeError(
                f"Exact resume requires a positive per-rank batch size, got {batch_size!r}"
            )
        if type(num_workers) is not int or num_workers < 0:
            raise RuntimeError(
                f"Exact resume requires non-negative num_workers, got {num_workers!r}"
            )
        if prefetch_factor is not None and (
            type(prefetch_factor) is not int or prefetch_factor <= 0
        ):
            raise RuntimeError(
                f"Invalid train-loader prefetch_factor: {prefetch_factor!r}"
            )
        if type(persistent_workers) is not bool:
            raise RuntimeError("Train-loader persistent_workers must be a bool")
        if type(drop_last) is not bool:
            raise RuntimeError("Train-loader drop_last must be a bool")

        try:
            torchdata_version = importlib_metadata.version("torchdata")
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "Exact resume requires the installed torchdata package version"
            ) from exc

        return {
            "world_size": int(self.world_size),
            "per_rank_batch_size": batch_size,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "drop_last": drop_last,
            "sampler": self._sampler_contract(),
            "torch_version": str(torch.__version__),
            "torchdata_version": str(torchdata_version),
            "components": self._component_contract(),
        }

    @staticmethod
    def _capture_python_rng_state() -> Dict[str, Any]:
        version, internal_state, gaussian = random.getstate()
        if not isinstance(internal_state, tuple):
            raise RuntimeError("Python RNG internal state must be a tuple")
        return {
            "version": int(version),
            "state": [int(value) for value in internal_state],
            "gaussian": None if gaussian is None else float(gaussian),
        }

    @staticmethod
    def _capture_numpy_rng_state() -> Dict[str, Any]:
        bit_generator, state, position, has_gaussian, cached_gaussian = (
            np.random.get_state()
        )
        if not isinstance(state, np.ndarray) or state.dtype != np.uint32:
            raise RuntimeError("NumPy global RNG state must be a uint32 array")
        return {
            "bit_generator": str(bit_generator),
            "state": [int(value) for value in state.tolist()],
            "position": int(position),
            "has_gaussian": int(has_gaussian),
            "cached_gaussian": float(cached_gaussian),
        }

    def _capture_rank_state(self) -> Dict[str, Any]:
        loader_state = self.train_loader.state_dict()
        if not isinstance(loader_state, Mapping):
            raise RuntimeError(
                "Training data loader state_dict() must return a mapping"
            )
        loader_state = self.train_loader.validate_resume_state(
            loader_state,
            expected_data_epoch=int(self.data_epoch),
            expected_rank=int(self.rank),
        )
        cuda_state = None
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA trainer cannot capture RNG without CUDA")
            cuda_state = torch.cuda.get_rng_state(self.device).detach().cpu().clone()
        return {
            "rank": int(self.rank),
            "python_rng_state": self._capture_python_rng_state(),
            "numpy_rng_state": self._capture_numpy_rng_state(),
            "torch_cpu_rng_state": torch.get_rng_state().detach().cpu().clone(),
            "torch_cuda_rng_state": cuda_state,
            "train_loader_state_dict": loader_state,
            "data_epoch": int(self.data_epoch),
        }

    def _gather_checkpoint_inputs(self) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        try:
            local_payload = {
                "rank": int(self.rank),
                "error": None,
                "contract": self._build_resume_contract(),
                "rank_state": self._capture_rank_state(),
            }
        except BaseException as exc:
            local_payload = {
                "rank": int(self.rank),
                "error": f"{type(exc).__name__}: {exc}",
                "contract": None,
                "rank_state": None,
            }

        if self.distributed:
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError(
                    "Distributed checkpoint capture requires an initialized process group"
                )
            gathered: List[Any] = [None] * self.world_size
            dist.all_gather_object(gathered, local_payload)
        else:
            gathered = [local_payload]

        errors = []
        for expected_rank, payload in enumerate(gathered):
            if not isinstance(payload, Mapping):
                errors.append(
                    f"rank {expected_rank}: invalid checkpoint-capture payload"
                )
                continue
            if payload.get("rank") != expected_rank:
                errors.append(
                    f"rank {expected_rank}: payload reports rank {payload.get('rank')!r}"
                )
            error = payload.get("error")
            if error is not None:
                errors.append(f"rank {expected_rank}: {error}")
        if errors:
            raise RuntimeError(
                "Collective checkpoint capture failed: " + "; ".join(errors)
            )

        contract = gathered[0]["contract"]
        for rank, payload in enumerate(gathered[1:], start=1):
            if payload["contract"] != contract:
                raise RuntimeError(
                    "Collective checkpoint capture found different resume contracts: "
                    f"rank 0 != rank {rank}"
                )
        return dict(contract), [dict(payload["rank_state"]) for payload in gathered]

    def _gather_rank_errors(self, local_error: Optional[str]) -> List[Optional[str]]:
        if not self.distributed:
            return [local_error]
        errors: List[Optional[str]] = [None] * self.world_size
        dist.all_gather_object(errors, local_error)
        return errors

    @staticmethod
    def _format_rank_errors(stage: str, errors: List[Optional[str]]) -> str:
        details = [
            f"rank {rank}: {error}"
            for rank, error in enumerate(errors)
            if error is not None
        ]
        return f"Collective resume {stage} failed: " + "; ".join(details)

    @staticmethod
    def _validate_python_rng_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, Mapping):
            raise RuntimeError("Rank Python RNG state must be a mapping")
        expected = {"version", "state", "gaussian"}
        if set(state) != expected:
            raise RuntimeError("Rank Python RNG state fields are invalid")
        version = state["version"]
        values = state["state"]
        gaussian = state["gaussian"]
        if type(version) is not int:
            raise RuntimeError("Rank Python RNG version must be an int")
        if not isinstance(values, list) or not values:
            raise RuntimeError("Rank Python RNG internal state must be a non-empty list")
        if any(type(value) is not int for value in values):
            raise RuntimeError("Rank Python RNG internal state values must be ints")
        if gaussian is not None and type(gaussian) is not float:
            raise RuntimeError("Rank Python RNG gaussian cache must be float or None")
        return {
            "version": version,
            "state": list(values),
            "gaussian": gaussian,
        }

    @staticmethod
    def _validate_numpy_rng_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, Mapping):
            raise RuntimeError("Rank NumPy RNG state must be a mapping")
        expected = {
            "bit_generator",
            "state",
            "position",
            "has_gaussian",
            "cached_gaussian",
        }
        if set(state) != expected:
            raise RuntimeError("Rank NumPy RNG state fields are invalid")
        bit_generator = state["bit_generator"]
        values = state["state"]
        position = state["position"]
        has_gaussian = state["has_gaussian"]
        cached_gaussian = state["cached_gaussian"]
        if not isinstance(bit_generator, str) or not bit_generator:
            raise RuntimeError("Rank NumPy bit-generator name must be a string")
        if not isinstance(values, list) or not values:
            raise RuntimeError("Rank NumPy RNG state must be a non-empty list")
        if any(
            type(value) is not int or value < 0 or value > 0xFFFFFFFF
            for value in values
        ):
            raise RuntimeError("Rank NumPy RNG state values must be uint32 integers")
        if type(position) is not int or position < 0:
            raise RuntimeError("Rank NumPy RNG position must be non-negative")
        if type(has_gaussian) is not int or has_gaussian not in (0, 1):
            raise RuntimeError("Rank NumPy has_gaussian must be 0 or 1")
        if type(cached_gaussian) is not float:
            raise RuntimeError("Rank NumPy cached_gaussian must be a float")
        return {
            "bit_generator": bit_generator,
            "state": list(values),
            "position": position,
            "has_gaussian": has_gaussian,
            "cached_gaussian": cached_gaussian,
        }

    @staticmethod
    def _validate_torch_rng_tensor(value: Any, field_name: str) -> torch.Tensor:
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.uint8
            or value.ndim != 1
            or value.numel() == 0
        ):
            raise RuntimeError(
                f"Rank {field_name} must be a non-empty one-dimensional uint8 tensor"
            )
        return value.detach().cpu().clone()

    def _validate_rank_state(
        self, state: Any, *, expected_rank: int
    ) -> Dict[str, Any]:
        if not isinstance(state, Mapping):
            raise RuntimeError(f"Checkpoint rank_states[{expected_rank}] must be a mapping")
        expected_fields = {
            "rank",
            "python_rng_state",
            "numpy_rng_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_state",
            "train_loader_state_dict",
            "data_epoch",
        }
        if set(state) != expected_fields:
            raise RuntimeError(
                f"Checkpoint rank_states[{expected_rank}] fields are invalid"
            )
        if state["rank"] != expected_rank:
            raise RuntimeError(
                f"Checkpoint rank_states[{expected_rank}] reports rank {state['rank']!r}"
            )
        loader_state = state["train_loader_state_dict"]
        if not isinstance(loader_state, Mapping):
            raise RuntimeError(
                f"Checkpoint rank_states[{expected_rank}] loader state must be a mapping"
            )
        data_epoch = state["data_epoch"]
        if type(data_epoch) is not int or data_epoch < 0:
            raise RuntimeError(
                f"Checkpoint rank_states[{expected_rank}] data_epoch must be non-negative"
            )
        loader_state = self.train_loader.validate_resume_state(
            loader_state,
            expected_data_epoch=data_epoch,
            expected_rank=expected_rank,
        )
        cuda_state = state["torch_cuda_rng_state"]
        if self.device.type == "cuda":
            cuda_state = self._validate_torch_rng_tensor(
                cuda_state, "torch_cuda_rng_state"
            )
        elif cuda_state is not None:
            raise RuntimeError(
                f"Checkpoint rank_states[{expected_rank}] has CUDA RNG for a CPU trainer"
            )
        return {
            "rank": expected_rank,
            "python_rng_state": self._validate_python_rng_state(
                state["python_rng_state"]
            ),
            "numpy_rng_state": self._validate_numpy_rng_state(
                state["numpy_rng_state"]
            ),
            "torch_cpu_rng_state": self._validate_torch_rng_tensor(
                state["torch_cpu_rng_state"], "torch_cpu_rng_state"
            ),
            "torch_cuda_rng_state": cuda_state,
            "train_loader_state_dict": loader_state,
            "data_epoch": data_epoch,
        }

    def _validate_resume_contract(self, contract: Any) -> Dict[str, Any]:
        if not isinstance(contract, Mapping):
            raise RuntimeError("Checkpoint resume_contract must be a mapping")
        current = self._build_resume_contract()
        if set(contract) != set(current):
            missing = sorted(set(current).difference(contract))
            extra = sorted(set(contract).difference(current))
            raise RuntimeError(
                "Checkpoint resume contract fields differ from the current runtime: "
                f"missing={missing}, extra={extra}"
            )
        for field, expected in current.items():
            if contract[field] != expected:
                raise RuntimeError(
                    "Checkpoint resume contract does not match the current runtime: "
                    f"{field} checkpoint={contract[field]!r}, current={expected!r}"
                )
        return dict(contract)

    @staticmethod
    def _validate_component_state_presence(
        checkpoint: Mapping[str, Any], contract: Mapping[str, Any]
    ) -> None:
        components = contract.get("components")
        if not isinstance(components, Mapping):
            raise RuntimeError("Checkpoint component contract must be a mapping")
        expected_component_names = {
            "optimizer",
            "scheduler",
            "amp_scaler",
            "ema",
        }
        if set(components) != expected_component_names:
            raise RuntimeError("Checkpoint component contract fields are invalid")

        enabled = {}
        for name in expected_component_names:
            component = components[name]
            if not isinstance(component, Mapping):
                raise RuntimeError(
                    f"Checkpoint component contract {name} must be a mapping"
                )
            component_enabled = component.get("enabled")
            if type(component_enabled) is not bool:
                raise RuntimeError(
                    f"Checkpoint component contract {name}.enabled must be a bool"
                )
            enabled[name] = component_enabled

        expected_presence = {
            "model_state_dict": True,
            "optimizer_state_dict": enabled["optimizer"],
            "lr_scheduler_state_dict": enabled["scheduler"],
            "scaler_state_dict": enabled["amp_scaler"],
            "ema_state_dict": enabled["ema"],
        }
        for state_key, should_exist in expected_presence.items():
            exists = state_key in checkpoint
            if should_exist and not exists:
                raise RuntimeError(
                    f"Checkpoint format v3 is missing {state_key}; component "
                    "state presence must match the resume contract"
                )
            if not should_exist and exists:
                raise RuntimeError(
                    f"Checkpoint has unexpected {state_key}; component state "
                    "presence must match the resume contract"
                )

    def _restore_rng_state(self, state: Mapping[str, Any]) -> None:
        python_state = state["python_rng_state"]
        random.setstate(
            (
                python_state["version"],
                tuple(python_state["state"]),
                python_state["gaussian"],
            )
        )
        numpy_state = state["numpy_rng_state"]
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                np.asarray(numpy_state["state"], dtype=np.uint32),
                numpy_state["position"],
                numpy_state["has_gaussian"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(state["torch_cpu_rng_state"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["torch_cuda_rng_state"], self.device)

    def _model_state_target(self) -> nn.Module:
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    def _validate_resume_checkpoint(
        self, checkpoint: Any
    ) -> Dict[str, Any]:
        """Purely validate and normalize a full-state v3 checkpoint."""
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(
                "Full-state checkpoint must deserialize to a mapping, got "
                f"{type(checkpoint).__name__}"
            )
        artifact_kind = checkpoint.get("artifact_kind")
        if artifact_kind != "training_state":
            if artifact_kind == "model_best":
                raise RuntimeError(
                    "artifact_kind=model_best cannot be used for full-state resume; "
                    "load it through model.finetune_weights for a model-only warm start"
                )
            if artifact_kind is None:
                raise RuntimeError(
                    "Checkpoint artifact_kind is missing; checkpoint_format_version < 3 "
                    "or pre-artifact checkpoints cannot be used for full-state resume"
                )
            raise RuntimeError(
                "Full-state resume requires artifact_kind=training_state, got "
                f"{artifact_kind!r}"
            )
        if "checkpoint_format_version" not in checkpoint:
            raise RuntimeError(
                "Checkpoint checkpoint_format_version < 3 or missing; legacy "
                "checkpoints cannot be used for full-state resume. Load through "
                "model.finetune_weights for a model-only warm start instead."
            )

        version = self._require_checkpoint_int(
            checkpoint, "checkpoint_format_version"
        )
        if version != 3:
            raise RuntimeError(
                "Checkpoint checkpoint_format_version must equal 3 for full-state "
                f"resume, got {version}"
            )

        checkpoint_iteration_unit = checkpoint.get("iteration_unit")
        if checkpoint_iteration_unit is None:
            if self.iteration_unit != "legacy_micro_step":
                raise RuntimeError(
                    "Training-state checkpoint is missing iteration_unit. "
                    "Pre-unit checkpoints can only be resumed with explicit "
                    "solver.iteration_unit=legacy_micro_step; model-only warm-start "
                    "through model.finetune_weights remains allowed."
                )
            checkpoint_iteration_unit = "legacy_micro_step"
        if checkpoint_iteration_unit not in {
            "optimizer_step",
            "legacy_micro_step",
        }:
            raise RuntimeError(
                "Checkpoint iteration_unit must be 'optimizer_step' or "
                f"'legacy_micro_step', got {checkpoint_iteration_unit!r}"
            )
        if checkpoint_iteration_unit != self.iteration_unit:
            raise RuntimeError(
                "Checkpoint iteration_unit does not match the current config: "
                f"checkpoint={checkpoint_iteration_unit}, "
                f"config={self.iteration_unit}"
            )

        if "resume_contract" not in checkpoint:
            raise RuntimeError("Checkpoint format v3 is missing resume_contract")
        resume_contract = self._validate_resume_contract(
            checkpoint["resume_contract"]
        )
        self._validate_component_state_presence(checkpoint, resume_contract)
        if "rank_states" not in checkpoint:
            raise RuntimeError("Checkpoint format v3 is missing rank_states")
        rank_states = checkpoint["rank_states"]
        if not isinstance(rank_states, list):
            raise RuntimeError("Checkpoint rank_states must be a list")
        if len(rank_states) != self.world_size:
            raise RuntimeError(
                "Checkpoint rank_states length does not match world_size: "
                f"checkpoint={len(rank_states)}, current={self.world_size}"
            )
        validated_rank_states = [
            self._validate_rank_state(state, expected_rank=rank)
            for rank, state in enumerate(rank_states)
        ]

        grad_accum_steps = self._require_checkpoint_int(
            checkpoint, "grad_accum_steps"
        )
        accum_count = self._require_checkpoint_int(checkpoint, "accum_count")
        iteration = self._require_checkpoint_int(checkpoint, "iter")
        micro_step = self._require_checkpoint_int(checkpoint, "micro_step")
        optimizer_step = self._require_checkpoint_int(checkpoint, "optimizer_step")
        amp_skipped_steps = self._require_checkpoint_int(
            checkpoint, "amp_skipped_steps"
        )
        consecutive_amp_skips = self._require_checkpoint_int(
            checkpoint, "consecutive_amp_skips"
        )

        if grad_accum_steps <= 0:
            raise RuntimeError("Checkpoint grad_accum_steps must be positive")
        if grad_accum_steps != self.grad_accum_steps:
            raise RuntimeError(
                "Checkpoint grad_accum_steps does not match the current config: "
                f"checkpoint={grad_accum_steps}, config={self.grad_accum_steps}"
            )
        for field_name, value in (
            ("accum_count", accum_count),
            ("iter", iteration),
            ("micro_step", micro_step),
            ("optimizer_step", optimizer_step),
            ("amp_skipped_steps", amp_skipped_steps),
            ("consecutive_amp_skips", consecutive_amp_skips),
        ):
            if value < 0:
                raise RuntimeError(
                    f"Checkpoint {field_name} must be non-negative, got {value}"
                )
        if accum_count != 0:
            raise RuntimeError(
                "Checkpoint was captured inside a gradient-accumulation window: "
                f"accum_count={accum_count}"
            )
        expected_iteration = (
            optimizer_step
            if checkpoint_iteration_unit == "optimizer_step"
            else micro_step
        )
        if iteration != expected_iteration:
            counter_name = (
                "optimizer_step"
                if checkpoint_iteration_unit == "optimizer_step"
                else "micro_step"
            )
            raise RuntimeError(
                f"Checkpoint iter and {counter_name} fields disagree: "
                f"iter={iteration}, {counter_name}={expected_iteration}"
            )
        if micro_step % grad_accum_steps != 0:
            raise RuntimeError(
                f"Checkpoint micro_step={micro_step} is not on a "
                f"grad_accum_steps={grad_accum_steps} boundary"
            )
        budget_step = (
            optimizer_step
            if checkpoint_iteration_unit == "optimizer_step"
            else micro_step
        )
        if budget_step > self.max_iter:
            raise RuntimeError(
                f"Checkpoint {checkpoint_iteration_unit}={budget_step} exceeds "
                f"max_iter={self.max_iter}"
            )
        optimizer_attempts = micro_step // grad_accum_steps
        if optimizer_step + amp_skipped_steps != optimizer_attempts:
            raise RuntimeError(
                "Checkpoint optimizer-step accounting is inconsistent: "
                f"optimizer_step={optimizer_step} + "
                f"amp_skipped_steps={amp_skipped_steps} != "
                f"micro_step/grad_accum_steps={optimizer_attempts}"
            )
        if consecutive_amp_skips > amp_skipped_steps:
            raise RuntimeError(
                "Checkpoint consecutive_amp_skips cannot exceed total "
                f"amp_skipped_steps: consecutive={consecutive_amp_skips}, "
                f"total={amp_skipped_steps}"
            )
        if consecutive_amp_skips > self.max_consecutive_amp_skips:
            raise RuntimeError(
                "Checkpoint consecutive_amp_skips exceeds the configured policy: "
                f"checkpoint={consecutive_amp_skips}, "
                f"limit={self.max_consecutive_amp_skips}"
            )

        if "best_metric" not in checkpoint:
            raise RuntimeError("Checkpoint format v3 is missing best_metric")
        best_metric = self._validate_checkpoint_metric(
            checkpoint["best_metric"], "best_metric"
        )
        early_stop_state = self._validate_early_stop_state(checkpoint)

        model_state = self._normalize_resume_model_state(
            self._require_checkpoint_mapping(checkpoint, "model_state_dict")
        )
        optimizer_state = self._require_checkpoint_mapping(
            checkpoint, "optimizer_state_dict"
        )
        scheduler_state = None
        if self.lr_scheduler is not None:
            scheduler_state = self._require_checkpoint_mapping(
                checkpoint, "lr_scheduler_state_dict"
            )
        scaler_state = None
        if self.amp_enabled:
            if self.scaler is None:
                raise RuntimeError("AMP is enabled but the trainer has no GradScaler")
            scaler_state = self._require_checkpoint_mapping(
                checkpoint, "scaler_state_dict"
            )
        ema_state = None
        if self.ema is not None:
            ema_state = self._require_checkpoint_mapping(
                checkpoint, "ema_state_dict"
            )

        return {
            "iteration_unit": checkpoint_iteration_unit,
            "micro_step": micro_step,
            "optimizer_step": optimizer_step,
            "amp_skipped_steps": amp_skipped_steps,
            "consecutive_amp_skips": consecutive_amp_skips,
            "best_metric": best_metric,
            "early_stop_state": early_stop_state,
            "resume_contract": resume_contract,
            "rank_state": validated_rank_states[self.rank],
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "lr_scheduler_state_dict": scheduler_state,
            "scaler_state_dict": scaler_state,
            "ema_state_dict": ema_state,
        }

    @classmethod
    def _snapshot_state_to_cpu(cls, value: Any) -> Any:
        if torch.is_tensor(value):
            return _CpuTensorSnapshot(value)
        if isinstance(value, Mapping):
            try:
                snapshot = value.__class__()
            except TypeError:
                snapshot = {}
            for key, item in value.items():
                snapshot[copy.deepcopy(key)] = cls._snapshot_state_to_cpu(item)
            if hasattr(value, "_metadata"):
                snapshot._metadata = copy.deepcopy(value._metadata)
            return snapshot
        if isinstance(value, list):
            return [cls._snapshot_state_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            items = [cls._snapshot_state_to_cpu(item) for item in value]
            if hasattr(value, "_fields"):
                return value.__class__(*items)
            return tuple(items)
        if isinstance(value, set):
            return {cls._snapshot_state_to_cpu(item) for item in value}
        return copy.deepcopy(value)

    @classmethod
    def _clone_checkpoint_value_to_cpu(cls, value: Any) -> Any:
        """Clone checkpoint values to CPU without retaining live parameter aliases."""
        if torch.is_tensor(value):
            return value.detach().to(device="cpu", copy=True)
        if isinstance(value, Mapping):
            try:
                cloned = value.__class__()
            except TypeError:
                cloned = {}
            for key, item in value.items():
                cloned[copy.deepcopy(key)] = cls._clone_checkpoint_value_to_cpu(item)
            if hasattr(value, "_metadata"):
                cloned._metadata = copy.deepcopy(value._metadata)
            return cloned
        if isinstance(value, list):
            return [cls._clone_checkpoint_value_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            items = [cls._clone_checkpoint_value_to_cpu(item) for item in value]
            if hasattr(value, "_fields"):
                return value.__class__(*items)
            return tuple(items)
        return copy.deepcopy(value)

    @classmethod
    def _materialize_state_snapshot(cls, value: Any) -> Any:
        if isinstance(value, _CpuTensorSnapshot):
            return value.tensor.to(value.device)
        if isinstance(value, Mapping):
            try:
                restored = value.__class__()
            except TypeError:
                restored = {}
            for key, item in value.items():
                restored[copy.deepcopy(key)] = cls._materialize_state_snapshot(item)
            if hasattr(value, "_metadata"):
                restored._metadata = copy.deepcopy(value._metadata)
            return restored
        if isinstance(value, list):
            return [cls._materialize_state_snapshot(item) for item in value]
        if isinstance(value, tuple):
            items = [cls._materialize_state_snapshot(item) for item in value]
            if hasattr(value, "_fields"):
                return value.__class__(*items)
            return tuple(items)
        if isinstance(value, set):
            return {cls._materialize_state_snapshot(item) for item in value}
        return copy.deepcopy(value)

    def _capture_resume_snapshot(self) -> Dict[str, Any]:
        return {
            "model_state_dict": self._snapshot_state_to_cpu(
                self._model_state_target().state_dict()
            ),
            "optimizer_state_dict": self._snapshot_state_to_cpu(
                self.optimizer.state_dict()
            ),
            "lr_scheduler_state_dict": (
                None
                if self.lr_scheduler is None
                else self._snapshot_state_to_cpu(self.lr_scheduler.state_dict())
            ),
            "scaler_state_dict": (
                None
                if self.scaler is None
                else self._snapshot_state_to_cpu(self.scaler.state_dict())
            ),
            "ema_state_dict": (
                None
                if self.ema is None
                else self._snapshot_state_to_cpu(self.ema.state_dict())
            ),
            "rank_state": self._capture_rank_state(),
            "trainer_state": {
                "current_iter": self.current_iter,
                "start_iter": self.start_iter,
                "start_epoch": self.start_epoch,
                "data_epoch": self.data_epoch,
                "loader_state_restored": self._loader_state_restored,
                "optimizer_step": self.optimizer_step,
                "amp_skipped_steps": self.amp_skipped_steps,
                "consecutive_amp_skips": self.consecutive_amp_skips,
                "accum_count": self._accum_count,
                "best_metric": self.best_metric,
                "early_stop_best_metric": self.early_stop_best_metric,
                "patience_counter": self._patience_counter,
                "should_stop": self.early_stop,
            },
        }

    def _restore_resume_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        self._model_state_target().load_state_dict(
            self._materialize_state_snapshot(snapshot["model_state_dict"]),
            strict=True,
        )
        self.optimizer.load_state_dict(
            self._materialize_state_snapshot(snapshot["optimizer_state_dict"])
        )
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(
                self._materialize_state_snapshot(
                    snapshot["lr_scheduler_state_dict"]
                )
            )
        if self.scaler is not None:
            self.scaler.load_state_dict(
                self._materialize_state_snapshot(snapshot["scaler_state_dict"])
            )
        if self.ema is not None:
            self.ema.load_state_dict(
                self._materialize_state_snapshot(snapshot["ema_state_dict"])
            )

        rank_state = snapshot["rank_state"]
        self.train_loader.load_state_dict(
            copy.deepcopy(rank_state["train_loader_state_dict"])
        )

        trainer_state = snapshot["trainer_state"]
        self.current_iter = trainer_state["current_iter"]
        self.start_iter = trainer_state["start_iter"]
        self.start_epoch = trainer_state["start_epoch"]
        self.data_epoch = trainer_state["data_epoch"]
        self._loader_state_restored = trainer_state["loader_state_restored"]
        self.optimizer_step = trainer_state["optimizer_step"]
        self.amp_skipped_steps = trainer_state["amp_skipped_steps"]
        self.consecutive_amp_skips = trainer_state["consecutive_amp_skips"]
        self._accum_count = trainer_state["accum_count"]
        self.best_metric = trainer_state["best_metric"]
        self.early_stop_best_metric = trainer_state["early_stop_best_metric"]
        self._patience_counter = trainer_state["patience_counter"]
        self.early_stop = trainer_state["should_stop"]
        self._restore_rng_state(rank_state)

    def save_best_model_artifact(
        self,
        metrics: Mapping[str, Any],
        *,
        evaluated_weight_source: str,
        evaluated_model_state_dict: Mapping[str, Any],
        raw_model_state_dict: Mapping[str, Any],
    ) -> None:
        """Save the best lightweight artifact for canonical segmentation AP."""
        if self._accum_count != 0:
            raise RuntimeError(
                "Cannot save model_best inside a gradient-accumulation window: "
                f"accum_count={self._accum_count}"
            )
        if evaluated_weight_source not in {"raw", "ema"}:
            raise ValueError(
                "evaluated_weight_source must be exactly 'raw' or 'ema', got "
                f"{evaluated_weight_source!r}"
            )
        if not isinstance(evaluated_model_state_dict, Mapping):
            raise TypeError("evaluated_model_state_dict must be a mapping")
        if not isinstance(raw_model_state_dict, Mapping):
            raise TypeError("raw_model_state_dict must be a mapping")

        primary_state = self._clone_checkpoint_value_to_cpu(
            evaluated_model_state_dict
        )
        raw_state = self._clone_checkpoint_value_to_cpu(raw_model_state_dict)
        artifact = {
            "artifact_kind": "model_best",
            "artifact_format_version": 2,
            "evaluated_weight_source": evaluated_weight_source,
            "model_state_dict": primary_state,
            "raw_model_state_dict": raw_state,
            "metrics": {key: float(value) for key, value in metrics.items()},
            "best_metric": self.best_metric,
            "iter": self._iteration_step(),
            "iteration_unit": self.iteration_unit,
            "micro_step": self.current_iter,
            "optimizer_step": self.optimizer_step,
            "amp_skipped_steps": self.amp_skipped_steps,
            "grad_accum_steps": self.grad_accum_steps,
            "early_stop_state": self._early_stop_state_dict(),
            "config": self.config,
        }
        if self.ema is not None:
            artifact["ema_state_dict"] = self._clone_checkpoint_value_to_cpu(
                self.ema.state_dict()
            )

        validate_best_model_artifact(artifact)

        best_path = self.output_dir / "best.pt"
        save_checkpoint(artifact, best_path)
        print(f"[Checkpoint] Saved best model artifact to {best_path}")

    def save_checkpoint(self) -> None:
        """Overwrite the single resumable training-state checkpoint."""
        if self._accum_count != 0:
            raise RuntimeError(
                "Cannot save a checkpoint in the middle of a gradient-accumulation "
                f"window: accum_count={self._accum_count}, "
                f"grad_accum_steps={self.grad_accum_steps}"
            )

        resume_contract, rank_states = self._gather_checkpoint_inputs()
        write_error = None
        if self.rank == 0:
            try:
                checkpoint = {
                    "artifact_kind": "training_state",
                    "checkpoint_format_version": 3,
                    "resume_contract": resume_contract,
                    "rank_states": rank_states,
                    "iter": self._iteration_step(),
                    "iteration_unit": self.iteration_unit,
                    "micro_step": self.current_iter,
                    "optimizer_step": self.optimizer_step,
                    "amp_skipped_steps": self.amp_skipped_steps,
                    "consecutive_amp_skips": self.consecutive_amp_skips,
                    "grad_accum_steps": self.grad_accum_steps,
                    "accum_count": self._accum_count,
                    "model_state_dict": self._model_state_target().state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_metric": self.best_metric,
                    "early_stop_state": self._early_stop_state_dict(),
                    "config": self.config,
                }

                if self.lr_scheduler is not None:
                    checkpoint["lr_scheduler_state_dict"] = (
                        self.lr_scheduler.state_dict()
                    )

                if self.scaler is not None:
                    checkpoint["scaler_state_dict"] = self.scaler.state_dict()

                if self.ema is not None:
                    checkpoint["ema_state_dict"] = self.ema.state_dict()

                save_checkpoint(checkpoint, self.output_dir / "last.pt")
            except BaseException as exc:
                write_error = f"{type(exc).__name__}: {exc}"

        if self.distributed:
            payload = [write_error if self.rank == 0 else None]
            dist.broadcast_object_list(payload, src=0, device=self.device)
            write_error = payload[0]
        if write_error is not None:
            raise RuntimeError(f"Rank-0 checkpoint write failed: {write_error}")

    def resume(self, checkpoint_path: str) -> None:
        """
        恢复训练。

        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.name != "last.pt":
            raise ValueError(
                "Full-state resume requires a checkpoint named exactly last.pt"
            )
        print(f"[Trainer] Resuming from {checkpoint_path}...")
        validation_error = None
        validation_exception = None
        validated = None
        try:
            checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
            validated = self._validate_resume_checkpoint(checkpoint)
        except BaseException as exc:
            validation_exception = exc
            validation_error = f"{type(exc).__name__}: {exc}"

        validation_errors = self._gather_rank_errors(validation_error)
        if any(error is not None for error in validation_errors):
            if not self.distributed and validation_exception is not None:
                raise validation_exception
            raise RuntimeError(
                self._format_rank_errors("validation", validation_errors)
            )
        assert validated is not None

        snapshot_error = None
        snapshot_exception = None
        previous_state = None
        try:
            previous_state = self._capture_resume_snapshot()
        except BaseException as exc:
            snapshot_exception = exc
            snapshot_error = f"{type(exc).__name__}: {exc}"
        snapshot_errors = self._gather_rank_errors(snapshot_error)
        if any(error is not None for error in snapshot_errors):
            if not self.distributed and snapshot_exception is not None:
                raise snapshot_exception
            raise RuntimeError(
                self._format_rank_errors("snapshot", snapshot_errors)
            )
        assert previous_state is not None

        # All v3 metadata, the runtime contract, and every rank-local state are
        # validated before any live training state is mutated.
        apply_error = None
        apply_exception = None
        try:
            self._model_state_target().load_state_dict(
                validated["model_state_dict"], strict=True
            )
            self.optimizer.load_state_dict(validated["optimizer_state_dict"])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(
                    validated["lr_scheduler_state_dict"]
                )
            if self.amp_enabled:
                self.scaler.load_state_dict(validated["scaler_state_dict"])
            if self.ema is not None:
                self.ema.load_state_dict(validated["ema_state_dict"])

            self.current_iter = validated["micro_step"]
            self.start_iter = validated["micro_step"]
            self.optimizer_step = validated["optimizer_step"]
            self.amp_skipped_steps = validated["amp_skipped_steps"]
            self.consecutive_amp_skips = validated["consecutive_amp_skips"]
            self._accum_count = 0
            self.best_metric = validated["best_metric"]
            self._apply_early_stop_state(validated["early_stop_state"])

            rank_state = validated["rank_state"]
            self.train_loader.load_state_dict(
                copy.deepcopy(rank_state["train_loader_state_dict"])
            )
            self.data_epoch = rank_state["data_epoch"]
            self.start_epoch = self.data_epoch
            self._loader_state_restored = True
        except BaseException as exc:
            apply_exception = exc
            apply_error = f"{type(exc).__name__}: {exc}"

        apply_errors = self._gather_rank_errors(apply_error)
        if any(error is not None for error in apply_errors):
            self._restore_resume_snapshot(previous_state)
            if not self.distributed and apply_exception is not None:
                raise apply_exception
            raise RuntimeError(self._format_rank_errors("state load", apply_errors))

        # RNG is restored strictly last.  No training iterator exists yet.
        rng_error = None
        rng_exception = None
        try:
            self._restore_rng_state(validated["rank_state"])
        except BaseException as exc:
            rng_exception = exc
            rng_error = f"{type(exc).__name__}: {exc}"

        rng_errors = self._gather_rank_errors(rng_error)
        if any(error is not None for error in rng_errors):
            self._restore_resume_snapshot(previous_state)
            if not self.distributed and rng_exception is not None:
                raise rng_exception
            raise RuntimeError(self._format_rank_errors("RNG load", rng_errors))

        if self.ema is not None:
            print("[Trainer] EMA state restored from checkpoint")

        print(
            f"[Trainer] Resumed from micro_step {self.start_iter}, "
            f"optimizer_step {self.optimizer_step}, best metric: {self.best_metric:.4f}")

    def _now_iso(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _now_console_ts(self) -> str:
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

    def _console_log(self, msg: str) -> None:
        # Avoid breaking tqdm progress bar formatting.
        if self._pbar is not None and tqdm is not None:
            tqdm.write(msg)
        else:
            print(msg)

    def _current_peak_memory_mb(self) -> Optional[float]:
        if self.device.type != "cuda":
            return None
        peak_mb = float(torch.cuda.max_memory_allocated(self.device)) / (
            1024.0 * 1024.0
        )
        return self._validate_runtime_telemetry_value(
            "peak_memory_mb", peak_mb, non_negative=True
        )

    @staticmethod
    def _validate_runtime_telemetry_value(
        name: str,
        value: Optional[float],
        *,
        non_negative: bool = False,
    ) -> Optional[float]:
        if value is None:
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            raise FloatingPointError(
                f"Runtime telemetry {name} must be finite, got {numeric!r}"
            )
        if non_negative and numeric < 0.0:
            raise ValueError(
                f"Runtime telemetry {name} must be non-negative, got {numeric!r}"
            )
        return numeric

    def _current_runtime_telemetry(self) -> Dict[str, Any]:
        amp_scale = None if self.scaler is None else float(self.scaler.get_scale())
        telemetry: Dict[str, Any] = {
            "amp_scale": self._validate_runtime_telemetry_value(
                "amp_scale", amp_scale, non_negative=True
            ),
            "amp_scale_before_step": self._validate_runtime_telemetry_value(
                "amp_scale_before_step",
                self._last_amp_scale_before_step,
                non_negative=True,
            ),
            "amp_scale_after_step": self._validate_runtime_telemetry_value(
                "amp_scale_after_step",
                self._last_amp_scale_after_step,
                non_negative=True,
            ),
            "preclip_grad_norm": self._validate_runtime_telemetry_value(
                "preclip_grad_norm",
                self._last_preclip_grad_norm,
                non_negative=True,
            ),
            "grad_finite": self._last_grad_finite,
            "first_nonfinite_grad_param": self._last_nonfinite_grad_param,
            "nonfinite_grad_count": int(self._last_nonfinite_grad_count),
            "consecutive_amp_skips": int(self.consecutive_amp_skips),
            "cuda_memory_allocated_mb": None,
            "cuda_memory_reserved_mb": None,
            "cuda_max_memory_reserved_mb": None,
        }
        if self.device.type != "cuda":
            return telemetry

        mib = 1024.0 * 1024.0
        cuda_values = {
            "cuda_memory_allocated_mb": (
                float(torch.cuda.memory_allocated(self.device)) / mib
            ),
            "cuda_memory_reserved_mb": (
                float(torch.cuda.memory_reserved(self.device)) / mib
            ),
            "cuda_max_memory_reserved_mb": (
                float(torch.cuda.max_memory_reserved(self.device)) / mib
            ),
        }
        for name, value in cuda_values.items():
            telemetry[name] = self._validate_runtime_telemetry_value(
                name, value, non_negative=True
            )
        return telemetry

    @staticmethod
    def _format_hms(seconds: Optional[float]) -> str:
        if seconds is None:
            return "--:--:--"
        s = max(0, int(seconds))
        h = s // 3600
        m = (s % 3600) // 60
        s = s % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _format_sec(seconds: Optional[float]) -> str:
        if seconds is None:
            return "--"
        return f"{seconds:.2f}s"

    @staticmethod
    def _format_coco_metrics(metrics: Dict[str, float]) -> str:
        def row(kind: str) -> Optional[str]:
            labels = [
                "AP", "AP50", "AP75", "AP80", "AP85", "AP90", "AP95", "AP_H",
                "APs", "APm", "APl",
            ]
            keys = [f"{kind}_{label}" for label in labels]
            if not any(k in metrics for k in keys):
                return None
            vals = [metrics.get(k, float("nan")) for k in keys]
            return (
                f"{kind}: AP={vals[0]:.4f}  AP50={vals[1]:.4f}  AP75={vals[2]:.4f}  "
                f"AP80={vals[3]:.4f}  AP85={vals[4]:.4f}  AP90={vals[5]:.4f}  "
                f"AP95={vals[6]:.4f}  AP_H={vals[7]:.4f}  APs={vals[8]:.4f}  "
                f"APm={vals[9]:.4f}  APl={vals[10]:.4f}"
            )

        lines = []
        segm = row("segm")
        bbox = row("bbox")
        if segm is not None:
            lines.append(segm)
        if bbox is not None:
            lines.append(bbox)
        return "\n".join(lines) if lines else "(no coco metrics)"


# =============================================================================
# DDP 训练器
# =============================================================================
class DDPTrainer(Trainer):
    """分布式数据并行训练器"""

    def __init__(self, *args, **kwargs):
        """
        初始化 DDP 训练器。

        需要确保在使用前调用:
            dist.init_process_group(backend='nccl')
        """
        find_unused_parameters = bool(kwargs.pop("find_unused_parameters", True))
        resume_path = kwargs.pop("resume", None)
        super().__init__(*args, resume=None, **kwargs)

        self.distributed = True
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = torch.cuda.current_device() if torch.cuda.is_available() else self.rank

        # 包装模型为 DDP
        ddp_kwargs = {
            "find_unused_parameters": find_unused_parameters,
        }
        if self.device.type == "cuda":
            ddp_kwargs["device_ids"] = [self.local_rank]
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            **ddp_kwargs,
        )

        if resume_path is not None:
            self.resume(resume_path)

        print(
            f"[DDPTrainer] Initialized on rank {self.rank}/{self.world_size}")

    def _log_training(self, losses: Dict[str, torch.Tensor]) -> None:
        """只在主进程记录日志"""
        if self.rank == 0:
            super()._log_training(losses)

    def save_checkpoint(self) -> None:
        """Collect rank-local state on every process; rank 0 writes the file."""
        super().save_checkpoint()

    def save_best_model_artifact(
        self,
        metrics: Mapping[str, Any],
        *,
        evaluated_weight_source: str,
        evaluated_model_state_dict: Mapping[str, Any],
        raw_model_state_dict: Mapping[str, Any],
    ) -> None:
        """Save the best-model artifact only on rank 0."""
        if self.rank == 0:
            super().save_best_model_artifact(
                metrics,
                evaluated_weight_source=evaluated_weight_source,
                evaluated_model_state_dict=evaluated_model_state_dict,
                raw_model_state_dict=raw_model_state_dict,
            )

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """分布式评估"""
        ema = getattr(self, "ema", None)
        evaluated_weight_source = "ema" if ema is not None else "raw"
        evaluated_model_state = None
        raw_model_state = None
        if ema is not None:
            ema.apply_shadow(self.model)

        save_best_artifact = False
        log_dict = {}
        try:
            self.model.eval()
            # run_inference_evaluation gathers distributed predictions first, so rank 0
            # can reuse the exact same finalization path as the single-GPU evaluator.
            # That keeps logs, checkpoint selection, and metric summaries equivalent.
            # Verified on 2026-04-13: no supervised loss is computed during validation.
            category_ids = list(getattr(self.val_dataset, "category_ids", [])) or None
            result = run_inference_evaluation(
                self.model,
                self.val_loader,
                coco_gt=getattr(self.val_dataset, "coco", None),
                device=self.device,
                output_dir=self.output_dir,
                optimizer_step=self.optimizer_step,
                amp_enabled=self.eval_amp_enabled,
                category_ids=category_ids,
                iou_types=getattr(self, "eval_iou_types", ["bbox", "segm"]),
                max_images=getattr(self, "eval_max_images", None),
                max_dets=getattr(self, "eval_max_dets", 100),
            )

            rank_zero_error = None
            if self.rank == 0:
                try:
                    previous_best_metric = self.best_metric
                    log_dict = self._finalize_eval_result(result)
                    save_best_artifact = self.best_metric > previous_best_metric
                    if save_best_artifact:
                        evaluated_model_state = self._clone_checkpoint_value_to_cpu(
                            self._model_state_target().state_dict()
                        )
                        raw_model_state = self._clone_checkpoint_value_to_cpu(
                            ema.backup if ema is not None else evaluated_model_state
                        )
                except Exception as exc:
                    rank_zero_error = f"{type(exc).__name__}: {exc}"

            self._sync_early_stop_state(rank_zero_error)
        finally:
            if ema is not None:
                ema.restore(self.model)
            self.model.train()

        rank_zero_artifact_error = None
        if self.rank == 0 and save_best_artifact:
            try:
                self.save_best_model_artifact(
                    log_dict,
                    evaluated_weight_source=evaluated_weight_source,
                    evaluated_model_state_dict=evaluated_model_state,
                    raw_model_state_dict=raw_model_state,
                )
            except Exception as exc:
                rank_zero_artifact_error = f"{type(exc).__name__}: {exc}"
        self._sync_early_stop_state(rank_zero_artifact_error)

        return log_dict if self.rank == 0 else {}
