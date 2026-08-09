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

from magformer.config import load_config, parse_args, set_seed
from magformer.utils.depth_sanity import (
    compute_depth_sanity_report,
    should_abort_for_depth_sanity,
    write_depth_sanity_report,
)
from magformer.engine import Trainer, DDPTrainer


_VC_SUDA_STAGES = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cfg_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    elif hasattr(obj, "dict"):
        obj = obj.dict()
    elif not isinstance(obj, dict) and hasattr(obj, "__dict__"):
        obj = vars(obj)
    if isinstance(obj, dict):
        return {k: _cfg_to_dict(v) if _is_config_like(v) else v for k, v in obj.items()}
    return obj


def _is_config_like(value: Any) -> bool:
    return isinstance(value, dict) or hasattr(value, "model_dump") or hasattr(value, "dict") or hasattr(value, "__dict__")


def is_vc_suda_enabled(config: Any) -> bool:
    vc_cfg = _cfg_get(config, "vc_suda", None)
    return bool(_cfg_get(vc_cfg, "enabled", False))


def _vc_suda_stage(config: Any) -> str:
    vc_cfg = _cfg_get(config, "vc_suda", None)
    return str(_cfg_get(vc_cfg, "stage", "A")).upper()


def _stage_at_least(stage: str, minimum: str) -> bool:
    return _VC_SUDA_STAGES.get(stage, -1) >= _VC_SUDA_STAGES[minimum]


def validate_vc_suda_config(config: Any) -> None:
    """Fail fast for VC-SUDA configs that cannot run correctly."""
    if not is_vc_suda_enabled(config):
        return

    vc_cfg = _cfg_get(config, "vc_suda", None)
    stage = _vc_suda_stage(config)
    if stage not in _VC_SUDA_STAGES:
        raise ValueError(f"Unsupported vc_suda.stage: {stage}")
    if not _cfg_get(vc_cfg, "source_ann", None):
        raise ValueError("vc_suda.source_ann is required when VC-SUDA is enabled")
    if _stage_at_least(stage, "B") and not _cfg_get(vc_cfg, "target_labeled_ann", None):
        raise ValueError("vc_suda.target_labeled_ann is required for VC-SUDA Stage B+")
    if _stage_at_least(stage, "C") and not _cfg_get(vc_cfg, "target_unlabeled_ann", None):
        raise ValueError("vc_suda.target_unlabeled_ann is required for VC-SUDA Stage C+")

    if _stage_at_least(stage, "C"):
        ema_cfg = _cfg_get(vc_cfg, "ema_teacher", None)
        if ema_cfg is None or not bool(_cfg_get(ema_cfg, "enabled", True)):
            raise ValueError("VC-SUDA Stage C+ requires vc_suda.ema_teacher.enabled=True")
        if _cfg_get(vc_cfg, "pseudo_label", None) is None:
            raise ValueError("VC-SUDA Stage C+ requires vc_suda.pseudo_label config")

    da_cfg = _cfg_get(vc_cfg, "domain_adaptation", None)
    if _stage_at_least(stage, "D"):
        if da_cfg is None:
            raise ValueError("VC-SUDA Stage D+ requires vc_suda.domain_adaptation config")
        if bool(_cfg_get(da_cfg, "use_uncertainty_weighting", False)):
            raise ValueError(
                "VC-SUDA uncertainty weighting is disabled until its parameters are explicitly "
                "added to the optimizer. Set use_uncertainty_weighting=False."
            )


def build_datasets(config):
    """
    Build training and validation datasets.

    VC-SUDA uses SemiSupervisedDataset for training and the ordinary COCO RGB-D
    dataset for validation. Ordinary training keeps the previous behavior.
    """
    data_cfg = config.data
    train_split = getattr(data_cfg, "train_split", "train")
    val_split = getattr(data_cfg, "val_split", "val")
    from magformer.data import CocoRgbdDataset

    # Build copy-paste config dict for CocoRgbdDataset (safe if attribute missing)
    _cp_cfg = getattr(data_cfg, "copy_paste", None)
    _copy_paste_dict = None
    if _cp_cfg is not None and getattr(_cp_cfg, "enabled", False):
        if hasattr(_cp_cfg, "model_dump"):
            _copy_paste_dict = _cp_cfg.model_dump()
        elif hasattr(_cp_cfg, "dict"):
            _copy_paste_dict = _cp_cfg.dict()
        else:
            _copy_paste_dict = vars(_cp_cfg)

    if is_vc_suda_enabled(config):
        validate_vc_suda_config(config)
        from magformer.data.semi_supervised_dataset import SemiSupervisedDataset

        vc_cfg = config.vc_suda
        dataset_root = data_cfg.dataset_root
        target_labeled_ann = _cfg_get(vc_cfg, "target_labeled_ann", None)
        target_unlabeled_ann = _cfg_get(vc_cfg, "target_unlabeled_ann", None)
        train_dataset = SemiSupervisedDataset(
            source_root=dataset_root,
            source_ann=_cfg_get(vc_cfg, "source_ann"),
            source_split=_cfg_get(vc_cfg, "source_split", train_split),
            source_transform=None,
            target_labeled_root=dataset_root if target_labeled_ann else None,
            target_labeled_ann=target_labeled_ann,
            target_labeled_split=_cfg_get(vc_cfg, "target_labeled_split", train_split),
            target_labeled_transform=None,
            target_unlabeled_root=dataset_root if target_unlabeled_ann else None,
            target_unlabeled_ann=target_unlabeled_ann,
            target_unlabeled_split=_cfg_get(vc_cfg, "target_unlabeled_split", train_split),
            weak_transform=None,
            strong_transform=None,
            stage=_vc_suda_stage(config),
        )
    else:
        train_dataset = CocoRgbdDataset(
            dataset_root=data_cfg.dataset_root,
            ann_file=data_cfg.train_ann,
            split=train_split,
            transform=None,
            is_train=True,
            copy_paste_config=_copy_paste_dict,
        )

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
    """Build train/validation data loaders."""
    from magformer.data.transforms import RGBDTransform
    from magformer.data.collate import collate_fn as ordinary_collate_fn
    from magformer.data.stateful import (
        EpochStatefulDistributedSampler,
        ExactStatefulDataLoader,
        StatefulRandomSampler,
    )

    rank = 0
    if is_distributed:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Distributed data loading requires an initialized process group"
            )
        rank = dist.get_rank()

    runtime_seed = int(config.runtime.seed)
    train_sampler_generator = torch.Generator()
    train_sampler_generator.manual_seed(runtime_seed + rank)
    train_worker_generator = torch.Generator()
    train_worker_generator.manual_seed(runtime_seed + 1_000_003 + rank)
    val_generator = torch.Generator()
    val_generator.manual_seed(runtime_seed + 2_000_006 + rank)

    # Resolve copy-paste config (safe if attribute missing on older configs)
    _cp_cfg = getattr(config.data, "copy_paste", None)
    _cp_enabled = bool(getattr(_cp_cfg, "enabled", False)) if _cp_cfg is not None else False

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
        copy_paste_enabled=_cp_enabled,
        copy_paste_prob=getattr(_cp_cfg, "prob", 0.5) if _cp_cfg is not None else 0.5,
        copy_paste_max_paste=getattr(_cp_cfg, "max_paste_instances", 5) if _cp_cfg is not None else 5,
        copy_paste_min_area=getattr(_cp_cfg, "min_instance_area", 16) if _cp_cfg is not None else 16,
        copy_paste_max_area_ratio=getattr(_cp_cfg, "max_instance_area_ratio", 0.3) if _cp_cfg is not None else 0.3,
        copy_paste_prefer_small=getattr(_cp_cfg, "prefer_small", True) if _cp_cfg is not None else True,
        copy_paste_scale_jitter=getattr(_cp_cfg, "scale_jitter", (0.8, 1.2)) if _cp_cfg is not None else (0.8, 1.2),
        copy_paste_iou_threshold=getattr(_cp_cfg, "iou_threshold", 0.7) if _cp_cfg is not None else 0.7,
        instance_bank=(
            train_dataset.instance_bank
            if _cp_enabled and hasattr(train_dataset, "instance_bank")
            else None
        ),
        sahi_crop_size=getattr(config.data, "sahi_crop_size", None),
    )

    train_collate_fn = ordinary_collate_fn
    if is_vc_suda_enabled(config):
        from magformer.data.semi_supervised_dataset import SemiSupervisedDataset

        if not isinstance(train_dataset, SemiSupervisedDataset):
            raise TypeError("VC-SUDA training requires SemiSupervisedDataset")
        train_dataset.source.transform = train_transform
        if train_dataset.target_labeled is not None:
            train_dataset.target_labeled.transform = train_transform
        train_dataset.weak_transform = train_transform
        train_dataset.strong_transform = train_transform
        train_collate_fn = SemiSupervisedDataset.collate_fn
    else:
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

    if is_distributed:
        train_sampler = EpochStatefulDistributedSampler(
            train_dataset,
            shuffle=True,
            seed=runtime_seed,
        )
        train_loader = ExactStatefulDataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=bool(num_workers),
            prefetch_factor=2 if num_workers else None,
            collate_fn=train_collate_fn,
            generator=train_worker_generator,
            snapshot_every_n_steps=1,
        )
    else:
        train_sampler = StatefulRandomSampler(
            train_dataset,
            generator=train_sampler_generator,
        )
        train_loader = ExactStatefulDataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=bool(num_workers),
            prefetch_factor=2 if num_workers else None,
            collate_fn=train_collate_fn,
            generator=train_worker_generator,
            snapshot_every_n_steps=1,
        )

    val_loader = None
    if val_dataset is not None:
        if is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            val_sampler = None
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=bool(num_workers),
            prefetch_factor=2 if num_workers else None,
            collate_fn=ordinary_collate_fn,
            generator=val_generator,
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


_CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_cpu_runtime(runtime_config: Any) -> None:
    """Apply process-wide CPU thread limits before data or model construction."""
    cpu_threads = int(runtime_config.cpu_threads)
    cpu_interop_threads = int(runtime_config.cpu_interop_threads)

    for variable in _CPU_THREAD_ENV_VARS:
        os.environ[variable] = str(cpu_threads)

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(cpu_interop_threads)

    applied_threads = int(torch.get_num_threads())
    applied_interop_threads = int(torch.get_num_interop_threads())
    if applied_threads != cpu_threads or applied_interop_threads != cpu_interop_threads:
        raise RuntimeError(
            "PyTorch CPU thread limits were not applied exactly: "
            f"requested intra_op={cpu_threads}, inter_op={cpu_interop_threads}; "
            f"applied intra_op={applied_threads}, inter_op={applied_interop_threads}"
        )

    print(
        "[Runtime] CPU thread limits: "
        f"intra_op={applied_threads}, inter_op={applied_interop_threads}"
    )


def initialize_runtime_device(runtime_config: Any, dist_ctx: Dict[str, Any]) -> torch.device:
    """Initialize the runtime device before DDP or model/data allocation."""
    runtime_device = str(runtime_config.device)
    if runtime_device == "cpu":
        print("[Runtime] CUDA allocator cap: disabled (device=cpu)")
        return torch.device("cpu")
    if runtime_device != "cuda":
        raise ValueError(
            f"runtime.device must be 'cuda' or 'cpu', got {runtime_device!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("runtime.device requests CUDA, but CUDA is not available")

    device_index = int(dist_ctx["device_index"])
    torch.cuda.set_device(device_index)

    memory_fraction = runtime_config.memory_fraction
    if memory_fraction is None:
        print(
            "[Runtime] CUDA allocator cap: "
            f"disabled (device=cuda:{device_index})"
        )
    else:
        memory_fraction = float(memory_fraction)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device_index)

        properties = torch.cuda.get_device_properties(device_index)
        total_memory_bytes = int(properties.total_memory)
        cap_memory_bytes = int(total_memory_bytes * memory_fraction)
        mib = 1024.0 * 1024.0
        print(
            "[Runtime] CUDA allocator cap: "
            f"device=cuda:{device_index}, fraction={memory_fraction:.4f}, "
            f"cap_mib={cap_memory_bytes / mib:.2f}, "
            f"total_mib={total_memory_bytes / mib:.2f}"
        )

    if bool(dist_ctx["is_distributed"]):
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")

    return torch.device(f"cuda:{device_index}")


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

    Moves model to CPU before loading to avoid CUDA context corruption when
    backbone shapes differ between checkpoint and model.

    Returns:
        包含 missing/unexpected key 的信息字典。
    """
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Finetune weights not found: {filepath}")

    # Ensure model is on CPU for safe weight loading
    original_device = next(model.parameters()).device
    if str(original_device) != "cpu":
        print(f"[Train] Moving model from {original_device} to CPU for weight loading")
        model = model.cpu()
        import torch as _torch
        _torch.cuda.empty_cache()

    from magformer.engine.utils import load_torch_checkpoint

    checkpoint = load_torch_checkpoint(filepath, map_location="cpu")

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
                if (ckpt_shape[0] < model_shape[0]
                        and len(ckpt_shape) == len(model_shape)
                        and all(ckpt_shape[d] == model_shape[d] for d in range(1, len(ckpt_shape)))):
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

    if bool(getattr(model, "residual_depth_fusion_enabled", False)):
        allowed_missing_prefixes = (
            "rgb_backbone.depth_encoder.",
            "rgb_backbone.depth_projections.",
        )
        allowed_unexpected_prefixes = (
            "depth_backbone.",
            "fusion.",
            "pixel_decoder.depth_pe.",
        )
        invalid_missing = [
            key
            for key in missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        invalid_unexpected = [
            key
            for key in unexpected_keys
            if not key.startswith(allowed_unexpected_prefixes)
        ]
        invalid_size_mismatches = [
            item
            for item in size_mismatch_keys
            if item[1] != "partial_copy"
        ]
        if invalid_missing or invalid_unexpected or invalid_size_mismatches:
            raise RuntimeError(
                "Residual depth fusion warm-start contains unapproved "
                "incompatibilities: "
                f"missing={invalid_missing}, unexpected={invalid_unexpected}, "
                f"size_mismatches={invalid_size_mismatches}"
            )

    print(f"[Train] Warm-start loaded model weights from {filepath}")
    print(f"[Train] Warm-start missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
    if missing_keys:
        print(f"[Train] Missing keys (first 20): {missing_keys[:20]}")
    if unexpected_keys:
        print(f"[Train] Unexpected keys (first 20): {unexpected_keys[:20]}")

    # Move model back to original device after loading
    model.to(original_device)

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
            if isinstance(module, torch.nn.ParameterDict):
                this_wd = 0.0
            if module_param_name in {"relative_position_bias_table", "absolute_pos_embed"}:
                this_wd = weight_decay_embed
            if module_param_name == "log_vars":
                this_wd = 0.0

            params.append(
                {
                    "params": [value],
                    "lr": lr,
                    "weight_decay": this_wd,
                }
            )

    # Uncertainty Weighting: add log_vars with weight_decay=0
    if hasattr(model, 'criterion') and hasattr(model.criterion, 'log_vars'):
        log_vars = model.criterion.log_vars
        if log_vars not in memo:
            params.append({
                "params": [log_vars],
                "lr": base_lr,
                "weight_decay": 0.0,
            })
            print(f"[Train] Added UW log_vars ({log_vars.shape[0]} tasks) to optimizer, lr={base_lr}, wd=0.0")

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


def _optimizer_steps_from_micro_steps(
    micro_steps: int,
    grad_accum_steps: int,
    *,
    field_name: str,
) -> int:
    micro_steps = int(micro_steps)
    grad_accum_steps = int(grad_accum_steps)
    if grad_accum_steps <= 0:
        raise ValueError(
            f"runtime.grad_accum_steps must be positive, got {grad_accum_steps}"
        )
    if micro_steps < 0:
        raise ValueError(f"{field_name} must be non-negative, got {micro_steps}")
    optimizer_steps, remainder = divmod(micro_steps, grad_accum_steps)
    if remainder != 0:
        raise ValueError(
            f"{field_name}={micro_steps} must be divisible by "
            f"runtime.grad_accum_steps={grad_accum_steps}"
        )
    return optimizer_steps


def _optimizer_schedule_value(
    configured_value: int,
    grad_accum_steps: int,
    *,
    iteration_unit: str,
    field_name: str,
) -> int:
    configured_value = int(configured_value)
    if configured_value < 0:
        raise ValueError(
            f"{field_name} must be non-negative, got {configured_value}"
        )
    if iteration_unit == "optimizer_step":
        return configured_value
    if iteration_unit == "legacy_micro_step":
        return _optimizer_steps_from_micro_steps(
            configured_value,
            grad_accum_steps,
            field_name=field_name,
        )
    raise ValueError(
        "solver.iteration_unit must be 'optimizer_step' or "
        f"'legacy_micro_step', got {iteration_unit!r}"
    )


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
    runtime_cfg = getattr(config, "runtime", None)
    grad_accum_steps = int(_cfg_get(runtime_cfg, "grad_accum_steps", 1))
    iteration_unit = str(
        getattr(solver_cfg, "iteration_unit", "optimizer_step")
    )
    max_optimizer_steps = _optimizer_schedule_value(
        solver_cfg.max_iter,
        grad_accum_steps,
        iteration_unit=iteration_unit,
        field_name="solver.max_iter",
    )
    warmup_optimizer_steps = _optimizer_schedule_value(
        getattr(solver_cfg, "warmup_iters", 0),
        grad_accum_steps,
        iteration_unit=iteration_unit,
        field_name="solver.warmup_iters",
    )
    warmup_factor = float(getattr(solver_cfg, "warmup_factor", 1.0))

    if scheduler_name == "poly":
        return build_warmup_poly_scheduler(
            optimizer=optimizer,
            max_iter=max_optimizer_steps,
            warmup_iters=warmup_optimizer_steps,
            warmup_factor=warmup_factor,
            warmup_method=warmup_method,
            power=0.9,
        )

    if scheduler_name in {"step", "multistep"}:
        optimizer_milestones = [
            _optimizer_schedule_value(
                milestone,
                grad_accum_steps,
                iteration_unit=iteration_unit,
                field_name=f"solver.steps[{index}]",
            )
            for index, milestone in enumerate(solver_cfg.steps)
        ]
        return build_warmup_multistep_scheduler(
            optimizer=optimizer,
            milestones=optimizer_milestones,
            gamma=float(solver_cfg.gamma),
            warmup_iters=warmup_optimizer_steps,
            warmup_factor=warmup_factor,
            warmup_method=warmup_method,
        )

    if scheduler_name == "cosine":
        return build_warmup_cosine_scheduler(
            optimizer=optimizer,
            max_iter=max_optimizer_steps,
            warmup_iters=warmup_optimizer_steps,
            warmup_factor=warmup_factor,
            warmup_method=warmup_method,
        )

    raise ValueError(f"Unknown lr_scheduler: {solver_cfg.lr_scheduler}")



def _build_vc_suda_components(config: Any, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    # Dead modules archived to .trash/dead_modules/ -- VC-SUDA components removed.
    raise RuntimeError(
        "VC-SUDA components have been archived. See .trash/dead_modules/. "
        "Set vc_suda.enabled=False or remove vc_suda config."
    )


def build_trainer(
    config,
    model,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    val_dataset,
    device: torch.device,
    output_dir: str,
    is_distributed: bool,
    amp_enabled: bool,
):
    log_period = int(_cfg_get(config.runtime, "log_period", 10))
    common_kwargs = dict(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        config=_cfg_to_dict(config),
        device=device,
        output_dir=str(output_dir),
        max_iter=int(_cfg_get(config.solver, "max_iter", 0)),
        eval_period=int(_cfg_get(config.runtime, "eval_period", 0)),
        checkpoint_period=int(_cfg_get(config.runtime, "checkpoint_period", 0)),
        log_period=log_period,
        amp_enabled=amp_enabled,
        clip_gradients=bool(_cfg_get(config.solver, "clip_gradients", True)),
        clip_value=float(_cfg_get(config.solver, "clip_value", 1.0)),
        resume=_cfg_get(config.runtime, "resume", None),
        logger_config=_cfg_to_dict(_cfg_get(config.runtime, "logger", {})),
    )

    if is_vc_suda_enabled(config):
        components = _build_vc_suda_components(config, model, device)
        trainer_cls = globals().get("VCSUDADDPTrainer" if is_distributed else "VCSUDATrainer")
        if trainer_cls is None:
            from magformer.engine.vc_suda_trainer import VCSUDATrainer, VCSUDADDPTrainer
            trainer_cls = VCSUDADDPTrainer if is_distributed else VCSUDATrainer
        if is_distributed:
            common_kwargs["find_unused_parameters"] = bool(_cfg_get(config.runtime, "find_unused_parameters", False))
        return trainer_cls(**common_kwargs, **components)

    if is_distributed:
        return DDPTrainer(
            criterion=None,
            find_unused_parameters=bool(_cfg_get(config.runtime, "find_unused_parameters", False)),
            **common_kwargs,
        )
    return Trainer(criterion=None, **common_kwargs)

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

    configure_cpu_runtime(config.runtime)
    validate_vc_suda_config(config)

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

    device = initialize_runtime_device(config.runtime, dist_ctx)

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

    # Constructing the trainer applies full-state resume.  In DDP this happens
    # only after DistributedDataParallel and the rank-local StatefulDataLoader
    # both exist, and before any training iterator can be created.
    amp_enabled = bool(config.solver.amp_enabled and device.type == "cuda")
    trainer = build_trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        device=device,
        output_dir=str(output_dir),
        is_distributed=is_distributed,
        amp_enabled=amp_enabled,
    )

    # 打印模型信息
    from magformer.engine.utils import count_parameters

    num_params = count_parameters(model, trainable_only=True)
    print(f"[Train] Model parameters: {num_params:,}")

    depth_sanity_path = output_dir / "depth_sanity.json"
    if init_mode == "resume":
        print(
            "[Train] Skipping depth sanity preflight during exact resume; "
            "the training-loader cursor is checkpoint state."
        )
    elif getattr(config.runtime, "skip_depth_sanity", False):
        print("[Train] Skipping depth sanity preflight by configuration.")
    else:
        try:
            batch = next(iter(train_loader))
            if is_vc_suda_enabled(config):
                images = batch["source_images"].to(device)
                depths = batch["source_depths"].to(device)
                padding_masks = batch.get("source_padding_masks", None)
                depth_valid_masks = batch.get("source_depth_valid_masks", None)
                noise_masks = batch.get("source_noise_masks", None)
            else:
                images = batch["images"].to(device)
                depths = batch["depths"].to(device)
                padding_masks = batch.get("padding_masks", None)
                depth_valid_masks = batch.get("depth_valid_masks", None)
                noise_masks = batch.get("noise_masks", None)
            if padding_masks is not None:
                padding_masks = padding_masks.to(device)
            if noise_masks is not None:
                noise_masks = noise_masks.to(device)
            if depth_valid_masks is not None:
                depth_valid_masks = depth_valid_masks.to(device)

            report: Dict[str, Any]
            if hasattr(model, "collect_preflight_diagnostics"):
                was_training = model.training
                model.eval()
                diagnostics = model.collect_preflight_diagnostics(
                    images=images,
                    depths=depths,
                    padding_masks=padding_masks,
                    depth_valid_masks=depth_valid_masks,
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
