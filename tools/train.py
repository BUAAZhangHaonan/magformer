#!/usr/bin/env python3
"""
MAGFormer Training Script

纯 PyTorch 实现的训练脚本，支持 AMP、DDP 和多日志后端。

Usage:
    python train.py --config configs/magformer.yaml --dataset-root /path/to/eccd
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader

from magformer.config import load_config, parse_args, setup_device, set_seed
from magformer.data import CocoRgbdDataset
from magformer.engine import Trainer, DDPTrainer


def build_datasets(config):
    """
    构建训练和验证数据集。

    Args:
        config: MAGFormer 配置

    Returns:
        (train_dataset, val_dataset)
    """
    data_cfg = config.data

    # 训练集
    train_dataset = CocoRgbdDataset(
        dataset_root=data_cfg.dataset_root,
        ann_file=data_cfg.train_ann,
        split="train",
        transform=None,  # Transform 在 DataLoader 中应用
        is_train=True,
    )

    # 验证集
    val_dataset = CocoRgbdDataset(
        dataset_root=data_cfg.dataset_root,
        ann_file=data_cfg.val_ann,
        split="val",
        transform=None,
        is_train=False,
    )

    return train_dataset, val_dataset


def build_data_loaders(
    config,
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int,
    is_distributed: bool = False,
):
    """
    构建数据加载器。

    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        batch_size: 批大小
        num_workers: 工作进程数
        is_distributed: 是否分布式

    Returns:
        (train_loader, val_loader)
    """
    from magformer.data.transforms import RGBDTransform
    from magformer.data.collate import collate_fn

    # 创建数据变换
    train_transform = RGBDTransform(
        image_size=config.data.image_size,
        min_scale=config.data.min_scale,
        max_scale=config.data.max_scale,
        random_flip=config.data.random_flip,
        rgb_brightness=config.data.rgb_photo_aug.brightness,
        rgb_contrast=config.data.rgb_photo_aug.contrast,
        rgb_saturation=config.data.rgb_photo_aug.saturation,
        rgb_hue=config.data.rgb_photo_aug.hue,
        depth_scale=config.data.depth.scale,
        depth_shift=config.data.depth.shift,
        depth_clip_min=config.data.depth.clip_min,
        depth_clip_max=config.data.depth.clip_max,
        depth_norm=config.data.depth.norm,
        depth_gaussian_std=config.data.depth_noise.gaussian_std,
        depth_speckle_std=config.data.depth_noise.speckle_std,
        depth_drop_prob=config.data.depth_noise.drop_prob,
        depth_drop_val=config.data.depth_noise.drop_val,
        is_train=True,
    )

    # 设置变换
    train_dataset.transform = train_transform
    if val_dataset is not None:
        val_dataset.transform = RGBDTransform(
            image_size=config.data.image_size,
            min_scale=config.data.min_scale,
            max_scale=config.data.max_scale,
            random_flip="none",
            rgb_brightness=0.0,
            rgb_contrast=0.0,
            rgb_saturation=0.0,
            rgb_hue=0.0,
            depth_scale=config.data.depth.scale,
            depth_shift=config.data.depth.shift,
            depth_clip_min=config.data.depth.clip_min,
            depth_clip_max=config.data.depth.clip_max,
            depth_norm=config.data.depth.norm,
            is_train=False,
        )

    # 训练加载器
    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    # 验证加载器
    val_loader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # 推理时 batch_size=1
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    return train_loader, val_loader


def build_model(config, device: torch.device):
    """
    构建模型。

    Args:
        config: 模型配置
        device: 计算设备

    Returns:
        模型实例
    """
    from magformer.models import build_model as _build_model

    model = _build_model(config)

    # 加载预训练权重
    if config.model.weights is not None:
        from magformer.engine.utils import load_checkpoint

        load_checkpoint(config.model.weights, model, strict=False)

    model = model.to(device)
    return model


def build_optimizer(model, config):
    """
    构建优化器。

    Args:
        model: 模型
        config: 求解器配置

    Returns:
        优化器
    """
    solver_cfg = config.solver

    # 参数组
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    params = [
        {"params": backbone_params, "lr": solver_cfg.base_lr * solver_cfg.backbone_multiplier},
        {"params": other_params, "lr": solver_cfg.base_lr},
    ]

    # AdamW
    if solver_cfg.optimizer == "ADAMW":
        optimizer = torch.optim.AdamW(
            params,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=solver_cfg.weight_decay,
        )
    elif solver_cfg.optimizer == "ADAM":
        optimizer = torch.optim.Adam(
            params,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=solver_cfg.weight_decay,
        )
    elif solver_cfg.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params,
            momentum=0.9,
            weight_decay=solver_cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {solver_cfg.optimizer}")

    return optimizer


def build_lr_scheduler(optimizer, config):
    """
    构建学习率调度器。

    Args:
        optimizer: 优化器
        config: 求解器配置

    Returns:
        学习率调度器
    """
    solver_cfg = config.solver

    if solver_cfg.lr_scheduler == "poly":
        from torch.optim.lr_scheduler import LambdaLR

        max_iter = solver_cfg.max_iter
        warmup_iters = solver_cfg.warmup_iters
        warmup_factor = solver_cfg.warmup_factor

        def lr_lambda(step):
            if step < warmup_iters:
                alpha = float(step) / warmup_iters
                return (1 - alpha) * warmup_factor + alpha
            else:
                progress = float(step - warmup_iters) / (max_iter - warmup_iters)
                return (1 - progress) ** 0.9

        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        from torch.optim.lr_scheduler import MultiStepLR

        scheduler = MultiStepLR(
            optimizer,
            milestones=solver_cfg.steps,
            gamma=solver_cfg.gamma,
        )

    return scheduler


def main():
    """主训练函数"""
    # 解析参数
    args = parse_args()
    cli_overrides = {}
    if args.dataset_root is not None:
        cli_overrides.setdefault("data", {})["dataset_root"] = args.dataset_root
    if args.weights is not None:
        cli_overrides.setdefault("model", {})["weights"] = args.weights
    if args.output_dir is not None:
        cli_overrides.setdefault("runtime", {})["output_dir"] = args.output_dir
    if args.resume is not None:
        cli_overrides.setdefault("runtime", {})["resume"] = args.resume

    # 加载配置
    config = load_config(args.config, overrides=cli_overrides)

    # CLI 覆盖运行时参数
    if args.gpus is not None:
        config.runtime.gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if args.num_workers is not None:
        config.runtime.num_workers = int(args.num_workers)
    if args.seed is not None:
        config.runtime.seed = int(args.seed)

    # 设置设备
    device = setup_device(config.runtime)

    # 设置随机种子
    set_seed(config.runtime.seed)

    # 创建输出目录
    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建数据集
    print("[Train] Building datasets...")
    train_dataset, val_dataset = build_datasets(config)

    # 构建数据加载器
    print("[Train] Building data loaders...")
    num_gpus = len(config.runtime.gpus) if config.runtime.device != "cpu" else 1
    is_distributed = config.runtime.ddp_enabled and num_gpus > 1
    train_loader, val_loader = build_data_loaders(
        config,
        train_dataset,
        val_dataset,
        batch_size=max(1, config.solver.ims_per_batch // num_gpus),
        num_workers=config.runtime.num_workers,
        is_distributed=is_distributed,
    )

    # 构建模型
    print("[Train] Building model...")
    model = build_model(config, device)

    # 构建优化器
    print("[Train] Building optimizer...")
    optimizer = build_optimizer(model, config)

    # 构建学习率调度器
    print("[Train] Building LR scheduler...")
    lr_scheduler = build_lr_scheduler(optimizer, config)

    # 打印模型信息
    from magformer.engine.utils import count_parameters

    num_params = count_parameters(model, trainable_only=True)
    print(f"[Train] Model parameters: {num_params:,}")

    # 构建训练器
    log_period = int(getattr(config.runtime, "log_period", 10))

    if is_distributed:
        import torch.distributed as dist

        # 初始化进程组
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )

        trainer = DDPTrainer(
            model=model,
            criterion=None,  # 损失在模型内部计算
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            val_dataset=val_dataset,
            config=config.model_dump(),
            device=device,
            output_dir=str(output_dir),
            max_iter=config.solver.max_iter,
            eval_period=config.runtime.eval_period,
            checkpoint_period=config.runtime.checkpoint_period,
            log_period=log_period,
            amp_enabled=config.solver.amp_enabled,
            clip_gradients=config.solver.clip_gradients,
            clip_value=config.solver.clip_value,
            resume=config.runtime.resume,
            logger_config=config.runtime.logger.model_dump(),
        )
    else:
        trainer = Trainer(
            model=model,
            criterion=None,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            val_dataset=val_dataset,
            config=config.model_dump(),
            device=device,
            output_dir=str(output_dir),
            max_iter=config.solver.max_iter,
            eval_period=config.runtime.eval_period,
            checkpoint_period=config.runtime.checkpoint_period,
            log_period=log_period,
            amp_enabled=config.solver.amp_enabled,
            clip_gradients=config.solver.clip_gradients,
            clip_value=config.solver.clip_value,
            resume=config.runtime.resume,
            logger_config=config.runtime.logger.model_dump(),
        )

    # 开始训练
    print("[Train] Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[Train] Training interrupted by user")
    finally:
        trainer.logger.close()


if __name__ == "__main__":
    main()
