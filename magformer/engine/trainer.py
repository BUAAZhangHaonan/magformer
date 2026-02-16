# -*- coding: utf-8 -*-
"""
MAGFormer Training Engine

纯 PyTorch 实现的通用训练器。
支持 AMP、DDP、Checkpoint 管理和 TensorBoard/WandB 日志。
"""

import os
import time
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler

from .utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    load_checkpoint,
    clip_gradients,
    get_lr,
    CombinedLogger,
)
from .evaluator import COCOEvaluator

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =============================================================================
# Trainer
# =============================================================================


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
        self.amp_enabled = amp_enabled
        self.clip_gradients = clip_gradients
        self.clip_value = clip_value
        self.metrics_log_file = self.output_dir / "metrics_log.jsonl"
        self.metrics_csv_file = self.output_dir / "metrics_log.csv"
        self.visualization_dir = self.output_dir / "visualizations"
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
        self.best_metric = 0.0

        # 设置日志
        self.logger = self._setup_logger(logger_config)

        # AMP Scaler
        self.scaler = GradScaler() if self.amp_enabled else None

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

        # 创建数据迭代器
        if self.distributed:
            sampler = self.train_loader.sampler
            sampler.set_epoch(self.start_epoch)

        data_iter = iter(self.train_loader)

        print(f"[Trainer] Starting training for {self.max_iter} iterations...")

        pbar = None
        if tqdm is not None:
            pbar = tqdm(total=self.max_iter, initial=self.current_iter, desc="MAGFormer Train", dynamic_ncols=True)

        while self.current_iter < self.max_iter:
            # 获取下一个批次
            try:
                batch = next(data_iter)
            except StopIteration:
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
                self.evaluate()

            # 保存检查点
            if (self.current_iter + 1) % self.checkpoint_period == 0:
                self.save_checkpoint(is_best=False)

        # 训练结束
        if pbar is not None:
            pbar.close()
        print("[Trainer] Training completed!")
        self.save_checkpoint(is_best=True)
        self.logger.close()

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        训练一个批次。

        Args:
            batch: 输入批次字典
        """
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
        if self.amp_enabled:
            with autocast('cuda'):
                outputs = self.model(
                    images,
                    depths,
                    targets,
                    padding_masks=padding_masks,
                    depth_noise_masks=noise_masks,
                )
                losses = self._compute_losses(outputs, targets)
        else:
            outputs = self.model(
                images,
                depths,
                targets,
                padding_masks=padding_masks,
                depth_noise_masks=noise_masks,
            )
            losses = self._compute_losses(outputs, targets)

        # 反向传播
        self.optimizer.zero_grad()

        if self.amp_enabled:
            self.scaler.scale(losses["total_loss"]).backward()
        else:
            losses["total_loss"].backward()

        # 梯度裁剪
        if self.clip_gradients:
            if self.amp_enabled:
                self.scaler.unscale_(self.optimizer)
            grad_norm = clip_gradients(self.model, self.clip_value)

        # 优化器步进
        optimizer_stepped = False
        if self.amp_enabled:
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_stepped = self.scaler.get_scale() >= prev_scale

            # AMP 下 GradScaler.step() 可能不会更新调度器检查所需的优化器状态标记
            # 这里在“确实发生参数更新”时显式同步，避免 lr_scheduler 的顺序假告警。
            if optimizer_stepped:
                if hasattr(self.optimizer, "_opt_called"):
                    self.optimizer._opt_called = True
                if hasattr(self.optimizer, "_step_count") and self.optimizer._step_count == 0:
                    self.optimizer._step_count = 1
        else:
            self.optimizer.step()
            optimizer_stepped = True

        if self.lr_scheduler is not None and optimizer_stepped:
            self.lr_scheduler.step()

        # 记录日志
        if self.current_iter % self.log_period == 0:
            self._log_training(losses)

        self.current_iter += 1
        return losses

    def _append_metrics_log(self, metrics: Dict[str, float], phase: str) -> None:
        payload = {
            "iter": int(self.current_iter),
            "phase": phase,
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
                raise ValueError("Model outputs must be a loss dict when criterion is None")
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

        # 打印进度
        loss_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in losses.items()])
        print(f"[Iter {self.current_iter}/{self.max_iter}] {loss_str}, LR: {lr:.6f}")

    def _save_eval_visualization(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, Any]) -> None:
        """保存 YOLOv8 风格的可视化结果"""
        predictions = outputs.get("predictions", None)
        if predictions is None or len(predictions) == 0:
            return

        import numpy as np
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
            save_path = self.visualization_dir / f"eval_iter_{self.current_iter:07d}_yolov8_empty.png"
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
        save_path = str(self.visualization_dir / f"eval_iter_{self.current_iter:07d}_yolov8.png")
        visualize_predictions(
            image=img,
            masks=masks,
            scores=scores,
            labels=labels,
            class_names=["component"],  # 单类别
            score_threshold=0.5,
            alpha=0.3,
            show_labels=False,
            show_contours=True,
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

        print(f"[Trainer] Evaluating at iteration {self.current_iter}...")

        self.model.eval()

        # 创建度量器
        meters = {"loss": AverageMeter()}

        # 创建 COCO 评估器
        coco_evaluator = None
        if self.val_dataset is not None and hasattr(self.val_dataset, 'coco'):
            coco_evaluator = COCOEvaluator(
                coco_gt=self.val_dataset.coco,
                iou_types=["bbox", "segm"],
                max_dets=100,
            )

        # 评估循环
        vis_saved = False
        total_preds = 0
        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            depths = batch["depths"].to(self.device)
            noise_masks = batch.get("noise_masks", None)
            if noise_masks is not None:
                noise_masks = noise_masks.to(self.device)
            padding_masks = batch.get("padding_masks", None)
            if padding_masks is not None:
                padding_masks = padding_masks.to(self.device)
            targets = batch.get("targets", None)
            image_ids = batch.get("image_ids", None)

            if targets is not None:
                targets = self._prepare_targets(targets, batch)

            # 前向传播
            if self.amp_enabled:
                with autocast('cuda'):
                    outputs = self.model(
                        images,
                        depths,
                        targets,
                        padding_masks=padding_masks,
                        depth_noise_masks=noise_masks,
                    )
            else:
                outputs = self.model(
                    images,
                    depths,
                    targets,
                    padding_masks=padding_masks,
                    depth_noise_masks=noise_masks,
                )

            if isinstance(outputs, dict) and "total_loss" in outputs:
                losses = self._compute_losses(outputs, targets)
                meters["loss"].update(losses["total_loss"].item())

            if not vis_saved and isinstance(outputs, dict):
                self._save_eval_visualization(batch, outputs)
                vis_saved = True

            # 收集预测结果用于 mAP 计算
            if coco_evaluator is not None and isinstance(outputs, dict):
                predictions = self._convert_to_coco_format(outputs, image_ids)
                total_preds += len(predictions)
                coco_evaluator.update(predictions)

        if coco_evaluator is not None:
            print(f"[Trainer] Total predictions collected: {total_preds}")

        # 记录结果
        avg_loss = meters["loss"].avg
        log_dict = {"val/loss": avg_loss}

        # 计算 mAP 指标
        if coco_evaluator is not None:
            coco_metrics = coco_evaluator.summarize()
            for key, value in coco_metrics.items():
                log_dict[f"val/{key}"] = value
            # 使用 segm_AP 作为主要指标
            if "segm_AP" in coco_metrics:
                log_dict["val/mAP"] = coco_metrics["segm_AP"]

        self.logger.log_scalars("val", log_dict, self.current_iter)
        self._append_metrics_log(log_dict, phase="val")

        loss_str = f"Loss {avg_loss:.4f}"
        if "val/mAP" in log_dict:
            loss_str += f", mAP {log_dict['val/mAP']:.4f}"
        print(f"[Trainer] Evaluation: {loss_str}")

        # 更新最佳模型
        if "val/mAP" in log_dict:
            metric = log_dict["val/mAP"]
            if metric > self.best_metric:
                self.best_metric = metric
                self.save_checkpoint(is_best=True)
        elif avg_loss < self.best_metric or self.best_metric == 0:
            # 如果没有 mAP，使用损失作为标准
            self.best_metric = avg_loss
            self.save_checkpoint(is_best=True)

        self.model.train()

        return log_dict

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
        predictions = outputs.get("predictions", None)
        if predictions is None:
            return []

        all_predictions = []
        import numpy as np
        for batch_idx, pred in enumerate(predictions):
            image_id = image_ids[batch_idx] if image_ids is not None else batch_idx
            scores = pred.get("scores", [])
            category_ids = pred.get("category_ids", [])
            masks = pred.get("masks", [])

            for i in range(len(scores)):
                score = float(scores[i]) if i < len(scores) else 0.0
                if score < 0.05:  # 过滤低置信度预测
                    continue

                category_id = int(category_ids[i]) + 1 if i < len(category_ids) else 1  # COCO类别从1开始

                mask = masks[i] if i < len(masks) else None
                if mask is None:
                    continue

                # 确保 mask 是二值掩码
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()

                # 处理 logits 或概率
                if mask.dtype != np.uint8:
                    # 如果值范围超过 [0, 1]，假设是 logits，需要 sigmoid
                    if mask.min() < 0 or mask.max() > 1:
                        mask = 1 / (1 + np.exp(-mask))  # sigmoid
                    binary_mask = (mask > 0.5).astype(np.uint8)
                else:
                    binary_mask = mask

                # 计算 bbox
                ys, xs = np.where(binary_mask > 0)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                # Internal convention: xyxy (exclusive max) to avoid ambiguity with COCO xywh.
                x1 = float(xs.min())
                y1 = float(ys.min())
                x2 = float(xs.max() + 1)
                y2 = float(ys.max() + 1)
                bbox = [x1, y1, x2, y2]

                all_predictions.append({
                    "image_id": int(image_id),
                    "category_id": category_id,
                    "score": score,
                    "mask": binary_mask,
                    "bbox": bbox,
                })

        return all_predictions

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

        filename = self.output_dir / f"checkpoint_iter_{self.current_iter:07d}.pth"
        save_checkpoint(checkpoint, filename, is_best=is_best)

    def resume(self, checkpoint_path: str) -> None:
        """
        恢复训练。

        Args:
            checkpoint_path: 检查点文件路径
        """
        print(f"[Trainer] Resuming from {checkpoint_path}...")
        checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer)

        self.start_iter = checkpoint.get("iter", 0)
        self.current_iter = self.start_iter
        self.best_metric = checkpoint.get("best_metric", 0.0)

        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"[Trainer] Resumed from iteration {self.start_iter}, best metric: {self.best_metric:.4f}")


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
        super().__init__(*args, **kwargs)

        self.distributed = True
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # 包装模型为 DDP
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.rank],
            find_unused_parameters=kwargs.get("find_unused_parameters", False),
        )

        print(f"[DDPTrainer] Initialized on rank {self.rank}/{self.world_size}")

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
        # 设置为 eval 模式
        self.model.eval()

        meters = {"loss": AverageMeter()}

        for batch in self.val_loader:
            images = batch["images"].to(self.rank)
            depths = batch["depths"].to(self.rank)
            noise_masks = batch.get("noise_masks", None)
            if noise_masks is not None:
                noise_masks = noise_masks.to(self.rank)
            padding_masks = batch.get("padding_masks", None)
            if padding_masks is not None:
                padding_masks = padding_masks.to(self.rank)
            targets = batch.get("targets", None)

            if targets is not None:
                targets = self._prepare_targets(targets, batch)

            # 前向传播
            if self.amp_enabled:
                with autocast('cuda'):
                    outputs = self.model(
                        images,
                        depths,
                        targets,
                        padding_masks=padding_masks,
                        depth_noise_masks=noise_masks,
                    )
            else:
                outputs = self.model(
                    images,
                    depths,
                    targets,
                    padding_masks=padding_masks,
                    depth_noise_masks=noise_masks,
                )

            if isinstance(outputs, dict) and "total_loss" in outputs:
                losses = self._compute_losses(outputs, targets)
                loss = losses["total_loss"].detach()
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                meters["loss"].update(loss.item() / self.world_size)

        # 记录结果 (仅主进程)
        if self.rank == 0:
            avg_loss = meters["loss"].avg
            self.logger.log_scalar("val/loss", avg_loss, self.current_iter)
            print(f"[DDPTrainer] Evaluation: Loss {avg_loss:.4f}")

        self.model.train()
