# -*- coding: utf-8 -*-
"""
MAGFormer Training Engine

纯 PyTorch 实现的通用训练器。
支持 AMP、DDP、Checkpoint 管理和 TensorBoard/WandB 日志。
"""

import time
import json
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .utils import (
    save_checkpoint,
    load_checkpoint,
    clip_gradients,
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


_CHECKPOINT_MAX_KEEP_UNSET = object()


def _as_cuda_amp(enabled: bool, device: torch.device) -> bool:
    """Return true only when CUDA AMP can actually run."""
    return bool(enabled and device.type == "cuda" and torch.cuda.is_available())


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
        checkpoint_max_keep: Any = _CHECKPOINT_MAX_KEEP_UNSET,
        log_period: int = 100,
        amp_enabled: bool = True,
        clip_gradients: bool = True,
        clip_value: float = 0.01,
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
            checkpoint_max_keep: 最多保留的 numbered checkpoints 数量；None 或 <=0 表示保留全部
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
        # Runtime config (needed early for eval and grad accum settings)
        if isinstance(self.config, dict):
            runtime_cfg = self.config.get("runtime", {})
        else:
            runtime_obj = getattr(self.config, "runtime", None)
            if hasattr(runtime_obj, "model_dump"):
                runtime_cfg = runtime_obj.model_dump()
            elif runtime_obj is not None:
                runtime_cfg = dict(vars(runtime_obj))
            else:
                runtime_cfg = {}
        if checkpoint_max_keep is _CHECKPOINT_MAX_KEEP_UNSET:
            checkpoint_max_keep = runtime_cfg.get("checkpoint_max_keep", 2)
        self.checkpoint_max_keep = checkpoint_max_keep
        # Early stopping is opt-in. A missing/null config must not create
        # implicit AP thresholds for full training runs.
        self.early_stop = False
        self._early_stop_enabled = False
        self._patience_counter = 0
        self._patience_limit = None
        self._min_delta = None
        self._target_ap = None
        es_cfg = runtime_cfg.get("early_stop")
        if isinstance(es_cfg, dict) and bool(es_cfg.get("enabled", False)):
            self._early_stop_enabled = True
            self._patience_limit = int(es_cfg.get("patience", 5))
            self._min_delta = float(es_cfg.get("min_delta", 0.1))
            self._target_ap = float(es_cfg.get("target_ap", 70.0))
        # Eval config: iou_types and max_images for faster eval during training
        self.eval_iou_types = runtime_cfg.get("eval_iou_types", None)
        self.eval_max_images = runtime_cfg.get("eval_max_images", None)
        self.eval_inference_topk = int(runtime_cfg.get("eval_inference_topk", 100))
        self.eval_max_dets = int(runtime_cfg.get("eval_max_dets", 100))
        self.eval_saves_best = bool(runtime_cfg.get("eval_saves_best", True))
        # Gradient accumulation
        self.grad_accum_steps = int(runtime_cfg.get("grad_accum_steps", 1))
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
        self.start_epoch = 0
        self.start_iter = 0
        self.current_iter = 0
        self.best_metric = float("-inf")
        self._train_start_monotonic: Optional[float] = None
        self._iter_time_window_sec = deque(maxlen=20)
        self._pbar = None

        # 设置日志
        self.logger = self._setup_logger(logger_config)

        # AMP Scaler
        self.scaler = GradScaler() if self.amp_enabled else None

        # EMA (Exponential Moving Average)
        self.ema = None
        if runtime_cfg.get("ema_enabled", False):
            self.ema = ModelEMA(
                self.model,
                decay=float(runtime_cfg.get("ema_decay", 0.9999)),
                warmup_iters=int(runtime_cfg.get("ema_warmup_iters", 200)),
            )
            print(f"[Trainer] EMA enabled: decay={runtime_cfg.get('ema_decay', 0.9999)}, warmup={runtime_cfg.get('ema_warmup_iters', 200)}")

        # 恢复训练
        if resume is not None:
            self.resume(resume)

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

    def train(self) -> None:
        """主训练循环"""
        self.model.train()

        if self._train_start_monotonic is None:
            self._train_start_monotonic = time.monotonic()
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

        # 创建数据迭代器
        if self.distributed:
            sampler = self.train_loader.sampler
            sampler.set_epoch(self.start_epoch)

        data_iter = iter(self.train_loader)

        self._console_log(
            f"[{self._now_console_ts()}] start iter={self.current_iter}/{self.max_iter} "
            f"eval_period={self.eval_period} ckpt_period={self.checkpoint_period} log_period={self.log_period}"
        )

        pbar = None
        if tqdm is not None:
            pbar = tqdm(total=self.max_iter, initial=self.current_iter,
                        desc="MAGFormer Train", dynamic_ncols=True)
        self._pbar = pbar

        while self.current_iter < self.max_iter:
            # 获取下一个批次
            try:
                batch = next(data_iter)
            except StopIteration:
                self.start_epoch += 1
                if self.distributed and hasattr(self.train_loader, 'sampler'):
                    self.train_loader.sampler.set_epoch(self.start_epoch)
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # 训练一个批次
            losses = self._train_step(batch)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "iter": self.current_iter,
                    "loss": f"{losses['total_loss'].item():.4f}",
                    "lr": f"{get_lr(self.optimizer):.6f}",
                })

            # 评估
            if (self.current_iter + 1) % self.eval_period == 0:
                eval_result = self.evaluate()
                # Early stopping
                if self._early_stop_enabled and not self.early_stop and eval_result:
                    current_ap = eval_result.get("val/mAP", 0.0)
                    # Early stopping decision only on rank 0
                    if self.rank == 0 or not self.distributed:
                        if current_ap >= self._target_ap:
                            self._console_log(
                                f"[EARLY STOP] AP={current_ap:.4f} >= target {self._target_ap}")
                            self.early_stop = True
                        elif current_ap > self.best_metric + self._min_delta:
                            self._patience_counter = 0
                        else:
                            self._patience_counter += 1
                            if self._patience_counter >= self._patience_limit:
                                self._console_log(
                                    f"[EARLY STOP] AP plateaued for {self._patience_limit} evals "
                                    f"(best={self.best_metric:.4f}, current={current_ap:.4f})")
                                self.early_stop = True
                # Broadcast early_stop decision to all DDP processes
                if self._early_stop_enabled and self.distributed and dist.is_available() and dist.is_initialized():
                    stop_tensor = torch.tensor([1 if self.early_stop else 0], device="cuda")
                    dist.broadcast(stop_tensor, src=0)
                    self.early_stop = bool(stop_tensor.item())
                if self.early_stop:
                    self._console_log(f"[EARLY STOP] Stopping at iter={self.current_iter}")
                    break

            # 保存检查点
            if (self.current_iter + 1) % self.checkpoint_period == 0:
                self.save_checkpoint(is_best=False)

        # 训练结束
        if pbar is not None:
            pbar.close()
        self._pbar = None
        self._console_log(f"[{self._now_console_ts()}] training completed")
        # Final evaluation if not already evaluated at this iteration
        if (self.current_iter + 1) % self.eval_period != 0:
            self._console_log(f"[{self._now_console_ts()}] running final evaluation at iter={self.current_iter}")
            self.evaluate()
        peak_memory_mb = self._current_peak_memory_mb()
        if peak_memory_mb is not None:
            self.peak_memory_file.write_text(
                f"{peak_memory_mb:.2f}\n", encoding="utf-8")
        # Best checkpoint is managed during evaluation; end-of-training checkpoint
        # should represent final state and must not overwrite model_best.pth.
        self.save_checkpoint(is_best=False)
        self.logger.close()

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        训练一个批次。

        Args:
            batch: 输入批次字典
        """
        iter_start = time.perf_counter()
        # 数据移到设备
        images = batch["images"].to(self.device)
        depths = batch["depths"].to(self.device)
        noise_masks = batch.get("noise_masks", None)
        if noise_masks is not None:
            noise_masks = noise_masks.to(self.device)
        padding_masks = batch.get("padding_masks", None)
        if padding_masks is not None:
            padding_masks = padding_masks.to(self.device)
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
                depth_noise_masks=noise_masks,
            )
            losses = self._compute_losses(outputs, targets)

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
            # 梯度裁剪
            if self.clip_gradients:
                if self.amp_enabled:
                    self.scaler.unscale_(self.optimizer)
                clip_gradients(self.model, self.clip_value)

            # 优化器步进
            optimizer_stepped = False
            if self.amp_enabled:
                prev_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_stepped = self.scaler.get_scale() >= prev_scale

                if optimizer_stepped:
                    if hasattr(self.optimizer, "_opt_called"):
                        self.optimizer._opt_called = True
                    if hasattr(self.optimizer, "_step_count") and self.optimizer._step_count == 0:
                        self.optimizer._step_count = 1
            else:
                self.optimizer.step()
                optimizer_stepped = True

            # Reset accumulation counter
            self._accum_count = 0

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # EMA update
            if self.ema is not None:
                self.ema.update(self.current_iter, self.model)

        iter_time_sec = time.perf_counter() - iter_start
        self._iter_time_window_sec.append(iter_time_sec)

        # 记录日志
        if self.current_iter % self.log_period == 0:
            self._log_training(losses)

        self.current_iter += 1
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
            iter_done = self.current_iter + (1 if phase == "train" else 0)
            remaining = max(0, self.max_iter - int(iter_done))
            eta_sec = iter_time_sec * remaining

        payload = {
            "iter": int(self.current_iter),
            "phase": phase,
            "wall_time": float(now_wall),
            "wall_time_iso": self._now_iso(),
            "elapsed_sec": float(elapsed_sec),
            "iter_time_sec": None if iter_time_sec is None else float(iter_time_sec),
            "eta_sec": None if eta_sec is None else float(eta_sec),
            "peak_memory_mb": self._current_peak_memory_mb(),
            **{k: float(v) for k, v in metrics.items()},
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
        if isinstance(result, dict):
            return result

        log_dict = result.log_dict
        coco_metrics = result.coco_metrics

        if result.visualization_batch is not None and result.visualization_outputs is not None:
            self._save_eval_visualization(result.visualization_batch, result.visualization_outputs)

        if log_dict:
            self.logger.log_scalars("val", log_dict, self.current_iter)
            self._append_metrics_log(log_dict, phase="val")

        summary = "no_metrics"
        if "val/mAP" in log_dict:
            summary = f"mAP={log_dict['val/mAP']:.4f}"
        elif "val/segm_AP" in log_dict:
            summary = f"segm_AP={log_dict['val/segm_AP']:.4f}"
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

        if "val/mAP" in log_dict and self.eval_saves_best:
            metric = log_dict["val/mAP"]
            if metric > self.best_metric:
                self.best_metric = metric
                self.save_checkpoint(is_best=True)
        elif "val/mAP" in log_dict:
            self._console_log(
                f"[{self._now_console_ts()}] eval_diagnostic_only "
                "runtime.eval_saves_best=false; skip model_best update"
            )

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

        self.logger.log_scalars("train", log_dict, self.current_iter)
        self._append_metrics_log(log_dict, phase="train")

        # Detectron2-like one-line progress
        iter_time_sec = None
        eta_sec = None
        if len(self._iter_time_window_sec) > 0:
            iter_time_sec = sum(self._iter_time_window_sec) / \
                len(self._iter_time_window_sec)
            remaining = max(0, self.max_iter - (self.current_iter + 1))
            eta_sec = iter_time_sec * remaining

        parts = [
            f"[{self._now_console_ts()}]",
            f"iter={self.current_iter}/{self.max_iter}",
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
                f"eval_iter_{self.current_iter:07d}_yolov8_empty.png"
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
                        f"eval_iter_{self.current_iter:07d}_yolov8.png")
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
            f"[{self._now_console_ts()}] eval iter={self.current_iter}")

        ema = getattr(self, "ema", None)
        if ema is not None:
            ema.apply_shadow(self.model)

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
                output_dir=self.output_dir,
                amp_enabled=self.amp_enabled,
                category_ids=category_ids,
                iou_types=getattr(self, "eval_iou_types", ["bbox", "segm"]),
                max_images=getattr(self, "eval_max_images", None),
                inference_topk=getattr(self, "eval_inference_topk", 100),
                max_dets=getattr(self, "eval_max_dets", 100),
            )
            return self._finalize_eval_result(result)
        finally:
            if ema is not None:
                ema.restore(self.model)
            self.model.train()

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
            score_threshold=0.05,
            mask_threshold=0.5,
            category_offset=1,
        )

    def save_checkpoint(self, is_best: bool = False) -> None:
        """
        保存检查点。

        Args:
            is_best: 是否是最佳模型
        """
        checkpoint = {
            "iter": self.current_iter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }

        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

        filename = self.output_dir / \
            f"checkpoint_iter_{self.current_iter:07d}.pth"
        save_checkpoint(checkpoint, filename, is_best=is_best)
        self._cleanup_old_checkpoints(max_keep=self.checkpoint_max_keep)

    def _cleanup_old_checkpoints(self, max_keep: Optional[int] = 2) -> None:
        """Keep only the most recent `max_keep` numbered checkpoints."""
        if self.output_dir is None:
            return
        if max_keep is None or max_keep <= 0:
            return
        ckpts = sorted(self.output_dir.glob("checkpoint_iter_*.pth"))
        while len(ckpts) > max_keep:
            ckpts[0].unlink(missing_ok=True)
            ckpts.pop(0)

    def resume(self, checkpoint_path: str) -> None:
        """
        恢复训练。

        Args:
            checkpoint_path: 检查点文件路径
        """
        print(f"[Trainer] Resuming from {checkpoint_path}...")
        checkpoint = load_checkpoint(
            checkpoint_path, self.model, self.optimizer)

        self.start_iter = checkpoint.get("iter", 0)
        self.current_iter = self.start_iter
        self.best_metric = checkpoint.get("best_metric", float("-inf"))

        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(
                checkpoint["lr_scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
            print("[Trainer] EMA state restored from checkpoint")

        print(
            f"[Trainer] Resumed from iteration {self.start_iter}, best metric: {self.best_metric:.4f}")

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
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        try:
            peak_bytes = torch.cuda.max_memory_allocated(self.device)
        except Exception:
            return None
        return float(peak_bytes) / (1024.0 * 1024.0)

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
            keys = [
                f"{kind}_AP",
                f"{kind}_AP50",
                f"{kind}_AP75",
                f"{kind}_APs",
                f"{kind}_APm",
                f"{kind}_APl",
            ]
            if not any(k in metrics for k in keys):
                return None
            vals = [metrics.get(k, float("nan")) for k in keys]
            return (
                f"{kind}: AP={vals[0]:.4f}  AP50={vals[1]:.4f}  AP75={vals[2]:.4f}  "
                f"APs={vals[3]:.4f}  APm={vals[4]:.4f}  APl={vals[5]:.4f}"
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
        super().__init__(*args, **kwargs)

        self.distributed = True
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = torch.cuda.current_device() if torch.cuda.is_available() else self.rank

        # Criterion class-weight buffers are constants; per-forward broadcasts
        # can mutate their autograd version across multi-forward steps.
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=False,
        )

        print(
            f"[DDPTrainer] Initialized on rank {self.rank}/{self.world_size}")

    def _log_training(self, losses: Dict[str, torch.Tensor]) -> None:
        """只在主进程记录日志"""
        if self.rank == 0:
            super()._log_training(losses)

    def save_checkpoint(self, is_best: bool = False) -> None:
        """只在主进程保存检查点"""
        if self.rank == 0:
            super().save_checkpoint(is_best=is_best)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """分布式评估"""
        ema = getattr(self, "ema", None)
        if ema is not None:
            ema.apply_shadow(self.model)

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
                amp_enabled=self.amp_enabled,
                category_ids=category_ids,
                iou_types=getattr(self, "eval_iou_types", ["bbox", "segm"]),
                max_images=getattr(self, "eval_max_images", None),
                inference_topk=getattr(self, "eval_inference_topk", 100),
                max_dets=getattr(self, "eval_max_dets", 100),
            )

            if self.rank == 0:
                self._finalize_eval_result(result)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            return result.log_dict if self.rank == 0 else {}
        finally:
            if ema is not None:
                ema.restore(self.model)
            self.model.train()
