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
from typing import Any, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader

from magformer.config import load_config, parse_args, setup_device, set_seed
from magformer.utils.depth_sanity import (
    compute_depth_sanity_report,
    should_abort_for_depth_sanity,
    write_depth_sanity_report,
)
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
    train_split = getattr(data_cfg, "train_split", "train")
    val_split = getattr(data_cfg, "val_split", "val")
    from magformer.data import CocoRgbdDataset

    # 训练集
    train_dataset = CocoRgbdDataset(
        dataset_root=data_cfg.dataset_root,
        ann_file=data_cfg.train_ann,
        split=train_split,
        transform=None,  # Transform 在 DataLoader 中应用
        is_train=True,
    )

    # 验证集
    val_dataset = CocoRgbdDataset(
        dataset_root=data_cfg.dataset_root,
        ann_file=data_cfg.val_ann,
        split=val_split,
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
        depth_per_sample_norm=getattr(config.data.depth, "per_sample_norm", True),
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
            depth_per_sample_norm=getattr(config.data.depth, "per_sample_norm", True),
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

        load_checkpoint(config.model.weights, model, strict=True)

    model = model.to(device)
    return model


def resolve_distributed_context(
    ddp_enabled: bool,
    runtime_gpus: list[int],
    runtime_device: str,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    env = env or os.environ
    if runtime_device == "cpu" or not ddp_enabled:
        return {
            "is_distributed": False,
            "requires_launcher": False,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "device_index": runtime_gpus[0] if runtime_gpus else 0,
        }

    requested_world_size = len(runtime_gpus) if runtime_gpus else 1
    env_world_size = int(env.get("WORLD_SIZE", "1"))
    if requested_world_size > 1 and env_world_size <= 1:
        return {
            "is_distributed": False,
            "requires_launcher": True,
            "world_size": requested_world_size,
            "rank": 0,
            "local_rank": 0,
            "device_index": runtime_gpus[0] if runtime_gpus else 0,
        }

    if env_world_size > 1:
        local_rank = int(env.get("LOCAL_RANK", env.get("RANK", "0")))
        rank = int(env.get("RANK", "0"))
        device_index = runtime_gpus[local_rank] if runtime_gpus and local_rank < len(runtime_gpus) else local_rank
        return {
            "is_distributed": True,
            "requires_launcher": False,
            "world_size": env_world_size,
            "rank": rank,
            "local_rank": local_rank,
            "device_index": device_index,
        }

    return {
        "is_distributed": False,
        "requires_launcher": False,
        "world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "device_index": runtime_gpus[0] if runtime_gpus else 0,
    }


def resolve_checkpoint_init_mode(resume: Optional[str], finetune_weights: Optional[str]) -> str:
    """
    解析训练初始化策略。

    优先级:
    1) resume: 恢复完整训练状态（模型 + optimizer + scheduler）
    2) finetune_weights: 仅 warm-start 模型参数
    3) none: 从头训练（或仅依赖 model.weights/backbone 预训练）
    """
    if resume:
        return "resume"
    if finetune_weights:
        return "finetune"
    return "none"


def _extract_model_state_dict(checkpoint_obj: Any) -> Dict[str, Any]:
    """从 checkpoint 对象中提取模型参数字典。"""
    if not isinstance(checkpoint_obj, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint_obj)}")

    if "model_state_dict" in checkpoint_obj:
        return checkpoint_obj["model_state_dict"]
    if "state_dict" in checkpoint_obj:
        return checkpoint_obj["state_dict"]
    return checkpoint_obj


def _strip_module_prefix_if_needed(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """兼容 DDP 保存的 `module.` 前缀参数名。"""
    if not state_dict:
        return state_dict
    if not all(isinstance(k, str) for k in state_dict.keys()):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_finetune_weights(
    model: torch.nn.Module,
    filename: str,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    加载 finetune warm-start 权重（仅模型参数，不恢复优化器/调度器状态）。

    Returns:
        包含 missing/unexpected key 的信息字典。
    """
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Finetune weights not found: {filepath}")

    from magformer.engine.utils import load_torch_checkpoint

    checkpoint = load_torch_checkpoint(
        filepath,
        map_location="cpu",
        verify_sha256=True,
    )

    state_dict = _extract_model_state_dict(checkpoint)
    state_dict = _strip_module_prefix_if_needed(state_dict)


    # Handle size-mismatched keys (e.g., query embeddings when num_queries changes)
    model_sd = model.state_dict()
    size_mismatch_keys = []
    for key in list(state_dict.keys()):
        if key in model_sd:
            ckpt_shape = state_dict[key].shape
            model_shape = model_sd[key].shape
            if ckpt_shape != model_shape:
                # Checkpoint smaller -> partial copy (e.g., 100 queries -> 200)
                if ckpt_shape[0] < model_shape[0] and len(ckpt_shape) == len(model_shape):
                    with torch.no_grad():
                        model_sd[key][:ckpt_shape[0]].copy_(state_dict[key])
                    size_mismatch_keys.append((key, "partial_copy", ckpt_shape, model_shape))
                else:
                    size_mismatch_keys.append((key, "skipped", ckpt_shape, model_shape))
                del state_dict[key]

    if size_mismatch_keys:
        print(f"[Train] Size-mismatched keys handled: {len(size_mismatch_keys)}")
        for key, action, ckpt_s, model_s in size_mismatch_keys:
            print(f"[Train]   {key}: {list(ckpt_s)} -> {list(model_s)} ({action})")
    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing_keys = list(getattr(incompatible, "missing_keys", [])) if incompatible is not None else []
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", [])) if incompatible is not None else []

    print(f"[Train] Warm-start loaded model weights from {filepath}")
    print(f"[Train] Warm-start missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
    if missing_keys:
        print(f"[Train] Missing keys (first 20): {missing_keys[:20]}")
    if unexpected_keys:
        print(f"[Train] Unexpected keys (first 20): {unexpected_keys[:20]}")

    return {
        "checkpoint_path": str(filepath),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


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

    base_lr = float(solver_cfg.base_lr)
    backbone_multiplier = float(solver_cfg.backbone_multiplier)
    rgb_backbone_multiplier = float(
        getattr(solver_cfg, "rgb_backbone_multiplier", backbone_multiplier)
        if getattr(solver_cfg, "rgb_backbone_multiplier", None) is not None
        else backbone_multiplier
    )
    depth_backbone_multiplier = float(
        getattr(solver_cfg, "depth_backbone_multiplier", backbone_multiplier * 2.0)
        if getattr(solver_cfg, "depth_backbone_multiplier", None) is not None
        else backbone_multiplier * 2.0
    )
    mgm_multiplier = float(getattr(solver_cfg, "mgm_multiplier", 2.0))
    weight_decay = float(solver_cfg.weight_decay)
    weight_decay_norm = float(getattr(solver_cfg, "weight_decay_norm", 0.0))
    weight_decay_embed = float(getattr(solver_cfg, "weight_decay_embed", 0.0))

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    # Detectron2-style per-parameter optimizer groups:
    # - backbone lr multiplier
    # - no/low weight decay for norm and embedding params
    params = []
    memo = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)

            full_name = f"{module_name}.{module_param_name}" if module_name else module_param_name
            module_name_l = module_name.lower()
            full_name_l = full_name.lower()

            lr = base_lr
            if "rgb_backbone" in module_name_l:
                lr = base_lr * rgb_backbone_multiplier
            elif "depth_backbone" in module_name_l:
                lr = base_lr * depth_backbone_multiplier
            elif (
                module_name_l.startswith("fusion")
                or "modality_fusion" in module_name_l
                or module_name_l.startswith("mgm")
                or ".mgm" in module_name_l
            ):
                lr = base_lr * mgm_multiplier
            elif "backbone" in full_name_l:
                # Compatibility fallback for non-standard backbone naming.
                lr = base_lr * backbone_multiplier
            this_wd = weight_decay

            if isinstance(module, norm_module_types):
                this_wd = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                this_wd = weight_decay_embed
            if module_param_name in {"relative_position_bias_table", "absolute_pos_embed"}:
                this_wd = weight_decay_embed

            params.append(
                {
                    "params": [value],
                    "lr": lr,
                    "weight_decay": this_wd,
                }
            )

    # AdamW
    optimizer_type = str(solver_cfg.optimizer).upper()
    if optimizer_type == "ADAMW":
        optimizer = torch.optim.AdamW(
            params,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=solver_cfg.weight_decay,
        )
    elif optimizer_type == "ADAM":
        optimizer = torch.optim.Adam(
            params,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=solver_cfg.weight_decay,
        )
    elif optimizer_type == "SGD":
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
    from magformer.engine.lr_scheduler import (
        build_warmup_cosine_scheduler,
        build_warmup_multistep_scheduler,
        build_warmup_poly_scheduler,
    )

    scheduler_name = str(getattr(solver_cfg, "lr_scheduler", "poly")).lower()
    warmup_method = str(getattr(solver_cfg, "warmup_method", "linear")).lower()
    warmup_iters = int(getattr(solver_cfg, "warmup_iters", 0))
    warmup_factor = float(getattr(solver_cfg, "warmup_factor", 1.0))

    if scheduler_name == "poly":
        return build_warmup_poly_scheduler(
            optimizer=optimizer,
            max_iter=int(solver_cfg.max_iter),
            warmup_iters=warmup_iters,
            warmup_factor=warmup_factor,
            warmup_method=warmup_method,
            power=0.9,
        )

    if scheduler_name in {"step", "multistep"}:
        return build_warmup_multistep_scheduler(
            optimizer=optimizer,
            milestones=list(solver_cfg.steps),
            gamma=float(solver_cfg.gamma),
            warmup_iters=warmup_iters,
            warmup_factor=warmup_factor,
            warmup_method=warmup_method,
        )

    if scheduler_name == "cosine":
        return build_warmup_cosine_scheduler(
            optimizer=optimizer,
            max_iter=int(solver_cfg.max_iter),
            warmup_iters=warmup_iters,
            warmup_factor=warmup_factor,
            warmup_method=warmup_method,
        )

    raise ValueError(f"Unknown lr_scheduler: {solver_cfg.lr_scheduler}")


def main():
    """主训练函数"""
    # 解析参数
    args = parse_args()
    cli_overrides = {}
    if args.dataset_root is not None:
        cli_overrides.setdefault("data", {})["dataset_root"] = args.dataset_root
    if args.weights is not None:
        cli_overrides.setdefault("model", {})["weights"] = args.weights
    if getattr(args, "finetune_weights", None) is not None:
        cli_overrides.setdefault("model", {})["finetune_weights"] = args.finetune_weights
    if args.output_dir is not None:
        cli_overrides.setdefault("runtime", {})["output_dir"] = args.output_dir
    if args.resume is not None:
        cli_overrides.setdefault("runtime", {})["resume"] = args.resume

    # 加载配置
    config = load_config(args.config, overrides=cli_overrides)

    # 验证配置（检查常见问题，如深度归一化）
    from magformer.config.validation import validate_config
    if not validate_config(config, strict=True):
        raise ValueError("Configuration validation failed; aborting training.")

    # CLI 覆盖运行时参数
    if args.gpus is not None:
        config.runtime.gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if args.num_workers is not None:
        config.runtime.num_workers = int(args.num_workers)
    if args.seed is not None:
        config.runtime.seed = int(args.seed)

    dist_ctx = resolve_distributed_context(
        ddp_enabled=bool(config.runtime.ddp_enabled),
        runtime_gpus=list(config.runtime.gpus),
        runtime_device=str(config.runtime.device),
    )
    if dist_ctx["requires_launcher"]:
        raise RuntimeError(
            "DDP is enabled with multiple GPUs, but torchrun/distributed environment is missing. "
            "Launch with torchrun --nproc_per_node=<num_gpus> ..."
        )

    if dist_ctx["is_distributed"]:
        torch.cuda.set_device(int(dist_ctx["device_index"]))
        device = torch.device(f"cuda:{int(dist_ctx['device_index'])}")
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
    else:
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
    num_gpus = int(dist_ctx["world_size"]) if config.runtime.device != "cpu" else 1
    is_distributed = bool(dist_ctx["is_distributed"])
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

    # 训练初始化策略：resume 优先于 finetune_weights
    init_mode = resolve_checkpoint_init_mode(
        resume=getattr(config.runtime, "resume", None),
        finetune_weights=getattr(config.model, "finetune_weights", None),
    )
    if init_mode == "finetune":
        finetune_path = getattr(config.model, "finetune_weights", None)
        print(f"[Train] Applying warm-start from model.finetune_weights: {finetune_path}")
        load_finetune_weights(model, finetune_path, strict=False)
    elif init_mode == "resume" and getattr(config.model, "finetune_weights", None):
        print(
            "[Train] runtime.resume is set; skip model.finetune_weights warm-start "
            "and restore full training state from resume checkpoint."
        )

    # 冻结指定模块（通过 config.model.freeze_modules 配置）
    freeze_modules = getattr(config.model, "freeze_modules", None)
    if freeze_modules:
        frozen_count = 0
        for prefix in freeze_modules:
            for name, param in model.named_parameters():
                if name.startswith(prefix) and param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1
        print(f"[Train] Frozen {frozen_count} parameters matching prefixes: {freeze_modules}")

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

    depth_sanity_path = output_dir / "depth_sanity.json"
    if getattr(config.runtime, "skip_depth_sanity", False):
        print("[Train] Skipping depth sanity preflight by configuration.")
    else:
        try:
            batch = next(iter(train_loader))
            images = batch["images"].to(device)
            depths = batch["depths"].to(device)
            padding_masks = batch.get("padding_masks", None)
            if padding_masks is not None:
                padding_masks = padding_masks.to(device)
            noise_masks = batch.get("noise_masks", None)
            if noise_masks is not None:
                noise_masks = noise_masks.to(device)

            report: Dict[str, Any]
            if hasattr(model, "collect_preflight_diagnostics"):
                was_training = model.training
                model.eval()
                diagnostics = model.collect_preflight_diagnostics(
                    images=images,
                    depths=depths,
                    padding_masks=padding_masks,
                    depth_noise_masks=noise_masks,
                )
                if was_training:
                    model.train()
                report = compute_depth_sanity_report(
                    depths=depths,
                    confidence_maps=diagnostics.get("confidence_maps"),
                    pred_masks=diagnostics.get("pred_masks"),
                )
            else:
                report = compute_depth_sanity_report(depths=depths)

            should_abort, reasons = should_abort_for_depth_sanity(report)
            report["should_abort"] = should_abort
            report["reasons"] = reasons
            write_depth_sanity_report(report, depth_sanity_path)
            print(f"[Train] Wrote depth sanity report to {depth_sanity_path}")
            if should_abort:
                print("[Train] Depth sanity preflight failed:")
                for reason in reasons:
                    print(f"[Train]   - {reason}")
                raise RuntimeError("Depth sanity preflight failed; aborting before full training.")
        except StopIteration:
            print("[Train] Depth sanity preflight skipped: empty train loader.")

    # 构建训练器
    log_period = int(getattr(config.runtime, "log_period", 10))

    amp_enabled = bool(config.solver.amp_enabled and device.type == "cuda")

    if is_distributed:
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
            amp_enabled=amp_enabled,
            clip_gradients=config.solver.clip_gradients,
            clip_value=config.solver.clip_value,
            resume=config.runtime.resume,
            logger_config=config.runtime.logger.model_dump(),
            find_unused_parameters=bool(config.runtime.find_unused_parameters),
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
            amp_enabled=amp_enabled,
            clip_gradients=config.solver.clip_gradients,
            clip_value=config.solver.clip_value,
            resume=config.runtime.resume,
            logger_config=config.runtime.logger.model_dump(),
        )

    if args.eval_only:
        print("[Train] Running evaluation only (--eval-only)")
        try:
            trainer.evaluate()
        finally:
            trainer.logger.close()
        return

    # 开始训练
    print("[Train] Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[Train] Training interrupted by user")
        raise
    finally:
        trainer.logger.close()


if __name__ == "__main__":
    main()
