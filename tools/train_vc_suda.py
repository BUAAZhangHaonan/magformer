#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VC-SUDA Training Entry Point

Usage:
    # Stage A: source pretrain
    python tools/train_vc_suda.py --config configs/vc_suda/stage_a_source_pretrain.yaml --gpus 0,1

    # Smoke test
    python tools/train_vc_suda.py --config configs/vc_suda/stage_a_source_pretrain.yaml --smoke-test --gpus 0
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import torch
import torch.distributed as dist

from magformer.config import load_config, setup_device, set_seed
from magformer.engine.utils import count_parameters


def parse_vc_suda_args():
    parser = argparse.ArgumentParser(description="VC-SUDA Training")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 iters then exit")
    parser.add_argument("--finetune-weights", type=str, default=None)
    return parser.parse_args()


def resolve_distributed_context(ddp_enabled, runtime_gpus, runtime_device, env=None):
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


def build_optimizer(model, config):
    """Build optimizer with Detectron2-style per-parameter groups."""
    solver_cfg = config.solver
    base_lr = float(solver_cfg.base_lr)
    backbone_multiplier = float(solver_cfg.backbone_multiplier)
    weight_decay = float(solver_cfg.weight_decay)
    weight_decay_norm = float(getattr(solver_cfg, "weight_decay_norm", 0.0))
    weight_decay_embed = float(getattr(solver_cfg, "weight_decay_embed", 0.0))

    norm_module_types = (
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm, torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm, torch.nn.LocalResponseNorm,
    )

    params = []
    memo = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad or value in memo:
                continue
            memo.add(value)
            full_name = f"{module_name}.{module_param_name}" if module_name else module_param_name
            module_name_l = module_name.lower()

            lr = base_lr
            if "rgb_backbone" in module_name_l:
                lr = base_lr * backbone_multiplier
            elif "depth_backbone" in module_name_l:
                lr = base_lr * backbone_multiplier * 2.0
            elif "fusion" in module_name_l or "modality_fusion" in module_name_l:
                lr = base_lr * 2.0
            elif "backbone" in module_name_l:
                lr = base_lr * backbone_multiplier

            this_wd = weight_decay
            if isinstance(module, norm_module_types):
                this_wd = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                this_wd = weight_decay_embed
            if module_param_name in {"relative_position_bias_table", "absolute_pos_embed"}:
                this_wd = weight_decay_embed

            params.append({"params": [value], "lr": lr, "weight_decay": this_wd})

    optimizer_type = str(solver_cfg.optimizer).upper()
    if optimizer_type == "ADAMW":
        return torch.optim.AdamW(params, betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_type == "ADAM":
        return torch.optim.Adam(params, betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_type == "SGD":
        return torch.optim.SGD(params, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {solver_cfg.optimizer}")


def build_lr_scheduler(optimizer, config):
    """Build LR scheduler."""
    from magformer.engine.lr_scheduler import build_warmup_poly_scheduler, build_warmup_multistep_scheduler

    solver_cfg = config.solver
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
    elif scheduler_name in {"step", "multistep"}:
        return build_warmup_multistep_scheduler(
            optimizer=optimizer,
            milestones=list(solver_cfg.steps),
            gamma=float(solver_cfg.gamma),
            warmup_iters=warmup_iters,
            warmup_factor=warmup_factor,
            warmup_method=warmup_method,
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {solver_cfg.lr_scheduler}")


def build_vc_suda_dataset(config):
    """Build SemiSupervisedDataset for VC-SUDA."""
    from magformer.data.semi_supervised_dataset import SemiSupervisedDataset
    from magformer.data.transforms import RGBDTransform, get_weak_augmentation

    vc_cfg = config.vc_suda
    data_cfg = config.data
    stage = vc_cfg.stage

    # Strong augmentation transform
    strong_transform = RGBDTransform(
        image_size=data_cfg.image_size,
        min_scale=data_cfg.min_scale,
        max_scale=data_cfg.max_scale,
        random_flip=data_cfg.random_flip,
        rgb_brightness=data_cfg.rgb_photo_aug.brightness,
        rgb_contrast=data_cfg.rgb_photo_aug.contrast,
        rgb_saturation=data_cfg.rgb_photo_aug.saturation,
        rgb_hue=data_cfg.rgb_photo_aug.hue,
        depth_scale=data_cfg.depth.scale,
        depth_shift=data_cfg.depth.shift,
        depth_clip_min=data_cfg.depth.clip_min,
        depth_clip_max=data_cfg.depth.clip_max,
        depth_norm=data_cfg.depth.norm,
        depth_per_sample_norm=getattr(data_cfg.depth, "per_sample_norm", True),
        is_train=True,
    )

    # Weak augmentation transform (for teacher pseudo-label generation)
    weak_transform = get_weak_augmentation(config)

    dataset = SemiSupervisedDataset(
        source_root=data_cfg.dataset_root,
        source_ann=vc_cfg.source_ann,
        source_transform=strong_transform,
        target_labeled_root=data_cfg.dataset_root,
        target_labeled_ann=vc_cfg.target_labeled_ann,
        target_unlabeled_root=data_cfg.dataset_root,
        target_unlabeled_ann=vc_cfg.target_unlabeled_ann,
        stage=stage,
        strong_transform=strong_transform,
        weak_transform=weak_transform,
    )

    return dataset


def build_vc_suda_criterion(config):
    """Build VCSUDACriterion wrapping SetCriterion."""
    from magformer.models.common.matcher import HungarianMatcher
    from magformer.models.common.criterion import SetCriterion
    from magformer.models.magformer.vc_suda_criterion import VCSUDACriterion

    vc_cfg = config.vc_suda
    mf_cfg = config.model.magformer.mask_former

    # Build Hungarian matcher
    matcher = HungarianMatcher(
        cost_class=float(mf_cfg.class_weight),
        cost_mask=float(mf_cfg.mask_weight),
        cost_dice=float(mf_cfg.dice_weight),
        num_points=int(mf_cfg.train_num_points),
    )

    # Build weight dict (including deep supervision aux losses)
    weight_dict = {
        "loss_ce": float(mf_cfg.class_weight),
        "loss_mask": float(mf_cfg.mask_weight),
        "loss_dice": float(mf_cfg.dice_weight),
    }
    if getattr(mf_cfg, "deep_supervision", False):
        num_aux = max(int(mf_cfg.dec_layers) - 1, 0)
        for i in range(num_aux):
            weight_dict.update({
                f"loss_ce_{i}": float(mf_cfg.class_weight),
                f"loss_mask_{i}": float(mf_cfg.mask_weight),
                f"loss_dice_{i}": float(mf_cfg.dice_weight),
            })

    # Build supervised SetCriterion
    supervised_criterion = SetCriterion(
        num_classes=int(config.model.magformer.sem_seg_head.num_classes),
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=float(mf_cfg.no_object_weight),
        losses=("labels", "masks"),
        num_points=int(mf_cfg.train_num_points),
        oversample_ratio=float(mf_cfg.oversample_ratio),
        importance_sample_ratio=float(mf_cfg.importance_sample_ratio),
    )

    # Build VCSUDACriterion wrapping SetCriterion
    criterion = VCSUDACriterion(
        supervised_criterion=supervised_criterion,
        pseudo_weight_ce=float(getattr(vc_cfg, "pseudo_weight_ce", 2.0)),
        pseudo_weight_mask=float(getattr(vc_cfg, "pseudo_weight_mask", 5.0)),
        pseudo_weight_dice=float(getattr(vc_cfg, "pseudo_weight_dice", 5.0)),
    )
    return criterion


def build_ema_teacher(model, config):
    """Build EMA teacher wrapper (only for stages C+)."""
    from magformer.models.common.ema_teacher import EMATeacherWrapper

    vc_cfg = config.vc_suda
    ema_cfg = vc_cfg.ema_teacher

    teacher = EMATeacherWrapper(
        student_model=model,
        momentum=ema_cfg.ema_momentum,
        warmup_steps=ema_cfg.warmup_steps,
    )
    print(f"[VCSUDA] EMA teacher created: momentum={ema_cfg.ema_momentum}, warmup={ema_cfg.warmup_steps}")
    return teacher


def build_pseudo_label_scorer(config):
    """Build pseudo-label quality scorer."""
    from magformer.models.common.pseudo_label_scorer import PseudoLabelScorer

    vc_cfg = config.vc_suda
    scorer = PseudoLabelScorer(
        max_instances=vc_cfg.pseudo_label.max_instances,
    )
    return scorer


def build_curriculum_scheduler(config):
    """Build curriculum scheduler for threshold decay."""
    from magformer.models.common.curriculum import CurriculumScheduler

    vc_cfg = config.vc_suda
    cur_cfg = vc_cfg.curriculum

    if not vc_cfg.pseudo_label.use_curriculum:
        return None

    scheduler = CurriculumScheduler(
        start_threshold=cur_cfg.start_threshold,
        end_threshold=cur_cfg.end_threshold,
        warmup_epochs=cur_cfg.warmup_epochs,
    )
    return scheduler


def build_domain_losses(config):
    """Build domain adaptation loss modules."""
    from magformer.models.magformer.domain_losses import (
        PrototypeAlignmentLoss,
        BoundaryConsistencyLoss,
        ModalityDropoutConsistencyLoss,
    )

    vc_cfg = config.vc_suda
    da_cfg = vc_cfg.domain_adaptation

    losses = {}
    if da_cfg.prototype_weight > 0:
        losses["prototype"] = PrototypeAlignmentLoss(
            num_classes=config.model.magformer.sem_seg_head.num_classes,
            feature_dim=config.model.magformer.sem_seg_head.convs_dim,
        )
    if da_cfg.boundary_weight > 0:
        losses["boundary"] = BoundaryConsistencyLoss(
            boundary_confidence_threshold=da_cfg.boundary_confidence_threshold,
        )
    if da_cfg.modality_dropout_weight > 0:
        losses["modality_dropout"] = ModalityDropoutConsistencyLoss()

    return losses


def main():
    args = parse_vc_suda_args()

    # Load config
    cli_overrides = {}
    if args.dataset_root:
        cli_overrides.setdefault("data", {})["dataset_root"] = args.dataset_root
    if args.output_dir:
        cli_overrides.setdefault("runtime", {})["output_dir"] = args.output_dir
    if args.resume:
        cli_overrides.setdefault("runtime", {})["resume"] = args.resume
    if args.finetune_weights:
        cli_overrides.setdefault("model", {})["finetune_weights"] = args.finetune_weights

    config = load_config(args.config, overrides=cli_overrides)

    # CLI overrides
    if args.gpus:
        config.runtime.gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if args.num_workers:
        config.runtime.num_workers = int(args.num_workers)
    if args.seed:
        config.runtime.seed = int(args.seed)

    vc_cfg = config.vc_suda
    stage = vc_cfg.stage
    print(f"[VCSUDA] Stage {stage} training")

    # Distributed context
    dist_ctx = resolve_distributed_context(
        ddp_enabled=bool(config.runtime.ddp_enabled),
        runtime_gpus=list(config.runtime.gpus),
        runtime_device=str(config.runtime.device),
    )
    if dist_ctx["requires_launcher"]:
        raise RuntimeError(
            "DDP enabled with multiple GPUs but torchrun environment missing. "
            "Launch with: torchrun --nproc_per_node=<num_gpus> tools/train_vc_suda.py ..."
        )

    if dist_ctx["is_distributed"]:
        torch.cuda.set_device(int(dist_ctx["device_index"]))
        device = torch.device(f"cuda:{int(dist_ctx['device_index'])}")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
    else:
        device = setup_device(config.runtime)

    set_seed(config.runtime.seed)
    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    print("[VCSUDA] Building dataset...")
    dataset = build_vc_suda_dataset(config)

    # Build data loader
    from torch.utils.data import DataLoader
    from magformer.data.semi_supervised_dataset import SemiSupervisedDataset

    is_distributed = bool(dist_ctx["is_distributed"])
    num_gpus = int(dist_ctx["world_size"]) if config.runtime.device != "cpu" else 1
    batch_size = max(1, config.solver.ims_per_batch // num_gpus)

    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True)
        train_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=config.runtime.num_workers, pin_memory=True,
            collate_fn=SemiSupervisedDataset.collate_fn,
        )
    else:
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=config.runtime.num_workers, pin_memory=True,
            collate_fn=SemiSupervisedDataset.collate_fn,
        )

    # For val: use standard COCO dataset
    val_loader = None
    val_dataset = None
    if config.data.val_ann:
        from magformer.data import CocoRgbdDataset
        from magformer.data.transforms import RGBDTransform
        from magformer.data.collate import collate_fn

        val_dataset = CocoRgbdDataset(
            dataset_root=config.data.dataset_root,
            ann_file=config.data.val_ann,
            split="val", transform=None, is_train=False,
        )
        val_dataset.transform = RGBDTransform(
            image_size=config.data.image_size,
            min_scale=config.data.min_scale, max_scale=config.data.max_scale,
            random_flip="none",
            rgb_brightness=0.0, rgb_contrast=0.0, rgb_saturation=0.0, rgb_hue=0.0,
            depth_scale=config.data.depth.scale, depth_shift=config.data.depth.shift,
            depth_clip_min=config.data.depth.clip_min, depth_clip_max=config.data.depth.clip_max,
            depth_norm=config.data.depth.norm,
            depth_per_sample_norm=getattr(config.data.depth, "per_sample_norm", True),
            is_train=False,
        )
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
        val_loader = DataLoader(
            val_dataset, batch_size=1, sampler=val_sampler,
            num_workers=config.runtime.num_workers, pin_memory=True,
            collate_fn=collate_fn,
        )

    # Estimate samples per epoch for curriculum scheduler
    samples_per_epoch = len(dataset)

    # Build model
    print("[VCSUDA] Building model...")
    from magformer.models import build_model as _build_model
    model = _build_model(config)
    model = model.to(device)

    # Load finetune weights
    if args.finetune_weights:
        from magformer.engine.utils import load_torch_checkpoint
        ckpt = load_torch_checkpoint(Path(args.finetune_weights), map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(f"[VCSUDA] Loaded finetune weights: missing={len(incompatible.missing_keys)}, "
              f"unexpected={len(incompatible.unexpected_keys)}")

    num_params = count_parameters(model, trainable_only=True)
    print(f"[VCSUDA] Model parameters: {num_params:,}")

    # Build VC-SUDA components based on stage
    ema_teacher = None
    pseudo_label_scorer = None
    curriculum_scheduler = None
    domain_losses = {}

    if stage in ("C", "D", "E"):
        ema_teacher = build_ema_teacher(model, config)
        pseudo_label_scorer = build_pseudo_label_scorer(config)
        curriculum_scheduler = build_curriculum_scheduler(config)

    if stage in ("D", "E"):
        domain_losses = build_domain_losses(config)

    # Build criterion
    criterion = build_vc_suda_criterion(config)

    # Build optimizer + scheduler
    print("[VCSUDA] Building optimizer + scheduler...")
    optimizer = build_optimizer(model, config)
    lr_scheduler = build_lr_scheduler(optimizer, config)

    # Add UncertaintyWeighting params to optimizer if enabled
    use_uncertainty_weighting = bool(vc_cfg.domain_adaptation.use_uncertainty_weighting)
    uw_module = None
    if use_uncertainty_weighting and stage in ("D", "E"):
        from magformer.models.magformer.domain_losses import UncertaintyWeighting
        uw_module = UncertaintyWeighting(3)
        uw_module = uw_module.to(device)
        optimizer.add_param_group({
            "params": list(uw_module.parameters()),
            "lr": float(config.solver.base_lr),
            "weight_decay": 0.0,
        })
        print("[VCSUDA] UncertaintyWeighting params added to optimizer (weight_decay=0)")

    amp_enabled = bool(config.solver.amp_enabled and device.type == "cuda")
    log_period = int(getattr(config.runtime, "log_period", 100))

    # VC-SUDA config dict for trainer
    vc_suda_config = vc_cfg.model_dump()

    # Set samples_per_epoch for curriculum epoch tracking
    vc_suda_config["_samples_per_epoch"] = samples_per_epoch

    # Build trainer
    from magformer.engine.vc_suda_trainer import VCSUDATrainer, VCSUDADDPTrainer

    trainer_kwargs = dict(
        model=model,
        criterion=criterion,
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
        ema_teacher=ema_teacher,
        pseudo_label_scorer=pseudo_label_scorer,
        curriculum_scheduler=curriculum_scheduler,
        vc_suda_config=vc_suda_config,
        domain_losses=domain_losses,
        use_uncertainty_weighting=use_uncertainty_weighting,
        uw_module=uw_module,
        smoke_test=args.smoke_test,
    )

    if is_distributed:
        trainer = VCSUDADDPTrainer(
            **trainer_kwargs,
            find_unused_parameters=bool(getattr(config.runtime, "find_unused_parameters", False)),
        )
    else:
        trainer = VCSUDATrainer(**trainer_kwargs)

    # Pass samples_per_epoch info
    trainer._samples_per_epoch = samples_per_epoch

    # Start training
    print(f"[VCSUDA] Starting training: stage={stage}, smoke_test={args.smoke_test}")
    try:
        trainer.train()
    except StopIteration:
        print("[VCSUDA] Smoke test completed")
    except KeyboardInterrupt:
        print("\n[VCSUDA] Training interrupted")
    finally:
        if hasattr(trainer, 'logger') and trainer.logger:
            trainer.logger.close()


if __name__ == "__main__":
    main()
