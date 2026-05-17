#!/usr/bin/env python3
"""
MAGFormer Training Script

纯 PyTorch 实现的训练脚本，支持 AMP、DDP 和多日志后端。

Usage:
    python train.py --config configs/magformer.yaml --dataset-root /path/to/eccd
"""

import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

# Add parent directory to path
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader

import yaml
from magformer.config import load_config, parse_args, setup_device, set_seed
from magformer.data.eval_subset import build_global_eval_subset
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


def is_vc_suda_offline_pseudo_enabled(config_or_vc_cfg: Any) -> bool:
    vc_cfg = _cfg_get(config_or_vc_cfg, "vc_suda", config_or_vc_cfg)
    offline_cfg = _cfg_get(vc_cfg, "offline_pseudo", None)
    return bool(_cfg_get(offline_cfg, "enabled", False))


def freeze_model_modules(model: torch.nn.Module, freeze_modules: Sequence[str]) -> Dict[str, Any]:
    matched_prefixes = {str(prefix): 0 for prefix in freeze_modules}
    frozen_count = 0
    for prefix in matched_prefixes:
        for name, param in model.named_parameters():
            if name.startswith(prefix):
                matched_prefixes[prefix] += 1
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1
    return {
        "matched_prefixes": matched_prefixes,
        "frozen_parameter_tensors": frozen_count,
    }


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
    source_datasets = _cfg_get(vc_cfg, "source_datasets", None)
    if source_datasets is None:
        if not _cfg_get(vc_cfg, "source_ann", None):
            raise ValueError("vc_suda.source_ann is required when VC-SUDA is enabled")
    elif len(source_datasets) == 0:
        raise ValueError("vc_suda.source_datasets must not be empty when set")
    if _stage_at_least(stage, "B") and not _cfg_get(vc_cfg, "target_labeled_ann", None):
        raise ValueError("vc_suda.target_labeled_ann is required for VC-SUDA Stage B+")
    if _stage_at_least(stage, "C") and not _cfg_get(vc_cfg, "target_unlabeled_ann", None):
        raise ValueError("vc_suda.target_unlabeled_ann is required for VC-SUDA Stage C+")

    if _stage_at_least(stage, "C"):
        offline_pseudo_enabled = is_vc_suda_offline_pseudo_enabled(vc_cfg)
        runtime_cfg = _cfg_get(config, "runtime", None)
        if bool(_cfg_get(runtime_cfg, "ema_enabled", False)):
            raise ValueError(
                "runtime.ema_enabled must be false for VC-SUDA Stage C+; "
                "use vc_suda.ema_teacher.enabled for pseudo-label teacher EMA only."
            )
        if not offline_pseudo_enabled:
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


def _sanitize_latest_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "run"


STAGE_LATEST_MARKER = "managed-by=magformer.tools.train.update_stage_latest_symlink.v1"


def stage_latest_marker_path(latest_path: Path | str) -> Path:
    latest = Path(latest_path)
    return latest.with_name(f".{latest.name}.managed")


def _is_marked_stage_latest_symlink(latest_path: Path) -> bool:
    marker = stage_latest_marker_path(latest_path)
    try:
        return marker.read_text(encoding="utf-8").strip() == STAGE_LATEST_MARKER
    except FileNotFoundError:
        return False


def _latest_symlink_resolves_to_output_parent(latest_path: Path) -> bool:
    try:
        return latest_path.resolve(strict=True).parent == latest_path.parent.resolve(strict=True)
    except FileNotFoundError:
        return False


def _assert_latest_path_replaceable(latest_path: Path) -> None:
    if not latest_path.exists() and not latest_path.is_symlink():
        return
    if not latest_path.is_symlink():
        raise FileExistsError(f"Refusing to replace non-symlink latest path: {latest_path}")
    if _latest_symlink_resolves_to_output_parent(latest_path):
        return
    if _is_marked_stage_latest_symlink(latest_path):
        return
    raise FileExistsError(f"Refusing to replace unmanaged latest symlink: {latest_path}")


def stage_latest_name(config: Any) -> Optional[str]:
    """Return the same-stage latest symlink name for a resolved config."""
    resolved = config
    name = _cfg_get(resolved, "name", None)
    if name is None and (hasattr(config, "model_dump") or hasattr(config, "dict")):
        resolved = _cfg_to_dict(config)
        name = _cfg_get(resolved, "name", None)

    if name:
        return f"{_sanitize_latest_component(str(name))}_latest"

    vc_cfg = _cfg_get(resolved, "vc_suda", None)
    if vc_cfg is None and resolved is not config:
        vc_cfg = _cfg_get(config, "vc_suda", None)
    if bool(_cfg_get(vc_cfg, "enabled", False)):
        stage = _sanitize_latest_component(str(_cfg_get(vc_cfg, "stage", "unknown")).lower())
        return f"vc_suda_stage_{stage}_latest"

    return None


def update_stage_latest_symlink(run_dir: Path | str, latest_name: str) -> Path:
    """Atomically point a same-stage latest symlink at run_dir.

    Existing real directories or files are never replaced. Existing symlinks are
    replaced by an atomic rename, so old run directories are not modified.
    """
    run_path = Path(run_dir).expanduser()
    if not run_path.exists() or not run_path.is_dir():
        raise FileNotFoundError(f"Run output directory does not exist: {run_path}")

    latest_path = run_path.parent / latest_name
    _assert_latest_path_replaceable(latest_path)

    temp_path = latest_path.with_name(f".{latest_path.name}.tmp.{os.getpid()}")
    if temp_path.exists() or temp_path.is_symlink():
        if not temp_path.is_symlink():
            raise FileExistsError(f"Refusing to replace non-symlink temporary path: {temp_path}")
        temp_path.unlink()

    target = os.path.relpath(run_path.resolve(), latest_path.parent.resolve())
    try:
        os.symlink(target, temp_path)
        _assert_latest_path_replaceable(latest_path)
        os.replace(temp_path, latest_path)
        stage_latest_marker_path(latest_path).write_text(
            STAGE_LATEST_MARKER + "\n",
            encoding="utf-8",
        )
    except Exception:
        if temp_path.exists() or temp_path.is_symlink():
            temp_path.unlink()
        raise

    return latest_path


def _git_text(args: Sequence[str], repo_root: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _collect_git_metadata(repo_root: Path) -> Dict[str, Any]:
    status = _git_text(["status", "--short"], repo_root)
    status_short = [] if status is None or status == "" else status.splitlines()
    return {
        "commit": _git_text(["rev-parse", "HEAD"], repo_root),
        "branch": _git_text(["branch", "--show-current"], repo_root),
        "status": status or "",
        "status_short": status_short,
        "dirty": bool(status_short),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


def write_run_provenance(
    *,
    config: Any,
    output_dir: Path | str,
    argv: Sequence[str],
    config_path: str,
    eval_only: bool,
    repo_root: Path | str,
    update_latest: bool = True,
) -> Optional[Path]:
    """Write resolved config and launch metadata for a future training/eval run."""
    out_path = Path(output_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    config_dict = _cfg_to_dict(config)
    config_text = yaml.safe_dump(
        config_dict,
        sort_keys=False,
        allow_unicode=True,
    )
    _atomic_write_text(out_path / "config_resolved.yaml", config_text)

    latest_path = None
    if update_latest:
        latest_name = stage_latest_name(config)
        if latest_name is not None:
            latest_path = update_stage_latest_symlink(out_path, latest_name=latest_name)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(argv),
        "command": shlex.join(str(item) for item in argv),
        "config_path": str(config_path),
        "output_dir": str(out_path),
        "eval_only": bool(eval_only),
        "git": _collect_git_metadata(Path(repo_root)),
        "latest_symlink": None if latest_path is None else str(latest_path),
    }
    _atomic_write_text(
        out_path / "run_metadata.json",
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return latest_path


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

    if is_vc_suda_enabled(config):
        validate_vc_suda_config(config)
        from magformer.data.semi_supervised_dataset import SemiSupervisedDataset

        vc_cfg = config.vc_suda
        dataset_root = data_cfg.dataset_root
        source_datasets = _cfg_get(vc_cfg, "source_datasets", None)
        if source_datasets is None:
            source_root = _cfg_get(vc_cfg, "source_root", None) or dataset_root
            source_ann = _cfg_get(vc_cfg, "source_ann")
        else:
            source_root = None
            source_ann = None
        target_labeled_ann = _cfg_get(vc_cfg, "target_labeled_ann", None)
        target_unlabeled_ann = _cfg_get(vc_cfg, "target_unlabeled_ann", None)
        target_unlabeled_sampling = _cfg_get(vc_cfg, "target_unlabeled_sampling", None)
        target_unlabeled_sampling_stats = None
        if bool(_cfg_get(target_unlabeled_sampling, "enabled", False)):
            target_unlabeled_sampling_stats = _cfg_get(target_unlabeled_sampling, "stats_path", None)
        offline_pseudo_config = _cfg_get(vc_cfg, "offline_pseudo", None)
        if hasattr(offline_pseudo_config, "model_dump"):
            offline_pseudo_config = offline_pseudo_config.model_dump()
        train_dataset = SemiSupervisedDataset(
            source_root=source_root,
            source_ann=source_ann,
            source_split=_cfg_get(vc_cfg, "source_split", train_split),
            source_datasets=source_datasets,
            source_transform=None,
            target_labeled_root=dataset_root if target_labeled_ann else None,
            target_labeled_ann=target_labeled_ann,
            target_labeled_split=_cfg_get(vc_cfg, "target_labeled_split", train_split),
            target_labeled_transform=None,
            target_unlabeled_root=dataset_root if target_unlabeled_ann else None,
            target_unlabeled_ann=target_unlabeled_ann,
            target_unlabeled_split=_cfg_get(vc_cfg, "target_unlabeled_split", train_split),
            target_unlabeled_sampling_stats=target_unlabeled_sampling_stats,
            offline_pseudo_config=offline_pseudo_config,
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

    train_collate_fn = ordinary_collate_fn
    if is_vc_suda_enabled(config):
        from magformer.data.semi_supervised_dataset import SemiSupervisedDataset

        if not isinstance(train_dataset, SemiSupervisedDataset):
            raise TypeError("VC-SUDA training requires SemiSupervisedDataset")
        train_dataset.set_source_transform(train_transform)
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
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_collate_fn,
        )

    val_loader = None
    if val_dataset is not None:
        eval_batch_size = int(getattr(getattr(config, "runtime", None), "eval_batch_size", 1))
        if eval_batch_size < 1:
            raise ValueError("runtime.eval_batch_size must be >= 1")
        val_dataset = build_global_eval_subset(
            val_dataset,
            getattr(getattr(config, "runtime", None), "eval_max_images", None),
        )
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=ordinary_collate_fn,
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


def _looks_like_model_state_dict(candidate: Dict[str, Any]) -> bool:
    """Return True for raw model state_dict-like mappings."""
    if not candidate:
        return False
    if not all(isinstance(k, str) for k in candidate.keys()):
        return False
    return any(torch.is_tensor(v) for v in candidate.values())


def _extract_model_state_dict(checkpoint_obj: Any) -> Dict[str, Any]:
    """从 checkpoint 对象中提取模型参数字典。"""
    if not isinstance(checkpoint_obj, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint_obj)}")

    for key in ("model_state_dict", "state_dict", "model"):
        if key not in checkpoint_obj:
            continue
        state_dict = checkpoint_obj[key]
        if not isinstance(state_dict, dict):
            raise TypeError(f"Checkpoint key '{key}' is not a state dict: {type(state_dict)}")
        if not _looks_like_model_state_dict(state_dict):
            raise ValueError(f"Checkpoint key '{key}' does not contain model tensor weights")
        return state_dict

    if _looks_like_model_state_dict(checkpoint_obj):
        return checkpoint_obj

    available = sorted(str(k) for k in checkpoint_obj.keys())
    raise ValueError(
        "Checkpoint does not contain model weights under any supported key "
        f"('model_state_dict', 'state_dict', 'model'); available keys: {available}"
    )


def _strip_module_prefix_if_needed(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """兼容 DDP 保存的 `module.` 前缀参数名。"""
    if not state_dict:
        return state_dict
    if not all(isinstance(k, str) for k in state_dict.keys()):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _validate_warm_start_load(
    *,
    matched_keys: set[str],
    expected_keys: set[str],
    unexpected_keys: list[str],
    min_match_ratio: float = 0.5,
) -> None:
    top_level_container_keys = {"model", "__author__", "source"}
    leaked_container_keys = sorted(top_level_container_keys.intersection(unexpected_keys))
    if leaked_container_keys:
        raise RuntimeError(
            "Warm-start checkpoint was not unpacked correctly; unexpected top-level "
            f"container keys reached model.load_state_dict: {leaked_container_keys}"
        )

    expected_count = len(expected_keys)
    matched_count = len(matched_keys)
    match_ratio = matched_count / expected_count if expected_count else 1.0
    if match_ratio < min_match_ratio:
        raise RuntimeError(
            "Warm-start matched too few expected model keys: "
            f"matched {matched_count}/{expected_count} expected keys "
            f"({match_ratio:.1%}); refusing to continue"
        )


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
    expected_keys = set(model_sd.keys())
    size_mismatch_keys = []
    partial_copy_keys = set()
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
                    partial_copy_keys.add(key)
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
    matched_keys = (set(state_dict.keys()).intersection(expected_keys)).union(partial_copy_keys)
    _validate_warm_start_load(
        matched_keys=matched_keys,
        expected_keys=expected_keys,
        unexpected_keys=unexpected_keys,
    )

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
        "matched_keys": len(matched_keys),
        "expected_keys": len(expected_keys),
        "match_ratio": len(matched_keys) / len(expected_keys) if expected_keys else 1.0,
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



def _build_vc_suda_components(config: Any, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    validate_vc_suda_config(config)
    vc_cfg = _cfg_get(config, "vc_suda")
    vc_dict = _cfg_to_dict(vc_cfg)
    stage = _vc_suda_stage(config)

    supervised_criterion = getattr(model, "criterion", None)
    if supervised_criterion is None:
        raise ValueError("VC-SUDA requires model.criterion for supervised loss")

    from magformer.models.magformer.vc_suda_criterion import VCSUDACriterion
    from magformer.models.common.curriculum import CurriculumScheduler
    from magformer.models.magformer.domain_losses import (
        BoundaryConsistencyLoss,
        ModalityDropoutConsistencyLoss,
        PrototypeAlignmentLoss,
    )

    ring_cfg = _cfg_get(vc_cfg, "pseudo_exterior_ring_loss", None)
    criterion = VCSUDACriterion(
        supervised_criterion=supervised_criterion,
        pseudo_unmatched_negative_enabled=bool(
            _cfg_get(vc_cfg, "pseudo_unmatched_negative_enabled", False)
        ),
        pseudo_unmatched_negative_weight=float(
            _cfg_get(vc_cfg, "pseudo_unmatched_negative_weight", 0.05)
        ),
        pseudo_unmatched_negative_score_thresh=float(
            _cfg_get(vc_cfg, "pseudo_unmatched_negative_score_thresh", 0.9)
        ),
        pseudo_exterior_ring_enabled=bool(_cfg_get(ring_cfg, "enabled", False)),
        pseudo_exterior_ring_weight=float(_cfg_get(ring_cfg, "weight", 0.0)),
        pseudo_exterior_ring_radius=int(_cfg_get(ring_cfg, "radius", 2)),
    )
    ema_teacher = None
    pseudo_label_scorer = None
    curriculum_scheduler = None
    if _stage_at_least(stage, "C"):
        offline_pseudo_enabled = is_vc_suda_offline_pseudo_enabled(vc_cfg)
        pl_cfg = _cfg_get(vc_cfg, "pseudo_label", None)
        if not offline_pseudo_enabled:
            ema_cfg = _cfg_get(vc_cfg, "ema_teacher")
            if isinstance(model, torch.nn.Module):
                from magformer.models.common.ema_teacher import EMATeacherWrapper
                ema_teacher = EMATeacherWrapper(
                    model,
                    momentum=float(_cfg_get(ema_cfg, "ema_momentum", _cfg_get(ema_cfg, "momentum", 0.999))),
                    warmup_steps=int(_cfg_get(ema_cfg, "warmup_steps", 500)),
                ).to(device)
            else:
                ema_teacher = object()
            from magformer.models.common.pseudo_label_scorer import PseudoLabelScorer
            pseudo_label_scorer = PseudoLabelScorer(max_instances=int(_cfg_get(pl_cfg, "max_instances", 100)))
        if pl_cfg is not None and bool(_cfg_get(pl_cfg, "use_curriculum", True)):
            cur_cfg = _cfg_get(vc_cfg, "curriculum", None)
            if cur_cfg is None:
                raise ValueError("VC-SUDA Stage C+ with curriculum requires vc_suda.curriculum config")
            curriculum_scheduler = CurriculumScheduler(
                start_threshold=float(_cfg_get(cur_cfg, "start_threshold", 0.7)),
                end_threshold=float(_cfg_get(cur_cfg, "end_threshold", 0.3)),
                warmup_epochs=int(_cfg_get(cur_cfg, "warmup_epochs", 15)),
            )

    da_cfg = _cfg_get(vc_cfg, "domain_adaptation", {})
    if bool(_cfg_get(da_cfg, "use_uncertainty_weighting", False)):
        raise ValueError(
            "VC-SUDA uncertainty weighting is disabled until its parameters are explicitly added to the optimizer."
        )
    domain_losses = {}
    if _stage_at_least(stage, "D"):
        if float(_cfg_get(da_cfg, "prototype_weight", 0.0)) > 0:
            domain_losses["prototype"] = PrototypeAlignmentLoss(
                num_classes=int(_cfg_get(da_cfg, "num_prototypes", 1)),
                feature_dim=int(_cfg_get(da_cfg, "feature_dim", 256)),
            ).to(device)
        if float(_cfg_get(da_cfg, "boundary_weight", 0.0)) > 0:
            domain_losses["boundary"] = BoundaryConsistencyLoss(
                boundary_confidence_threshold=float(_cfg_get(da_cfg, "boundary_confidence_threshold", 0.5))
            ).to(device)
        if float(_cfg_get(da_cfg, "modality_dropout_weight", 0.0)) > 0:
            domain_losses["modality_dropout"] = ModalityDropoutConsistencyLoss(
                dropout_prob=float(_cfg_get(da_cfg, "modality_dropout_prob", 0.3))
            ).to(device)

    return {
        "criterion": criterion,
        "ema_teacher": ema_teacher,
        "pseudo_label_scorer": pseudo_label_scorer,
        "curriculum_scheduler": curriculum_scheduler,
        "vc_suda_config": vc_dict,
        "domain_losses": domain_losses,
    }


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
        checkpoint_max_keep=_cfg_get(config.runtime, "checkpoint_max_keep", 2),
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

    if int(dist_ctx["rank"]) == 0:
        write_run_provenance(
            config=config,
            output_dir=output_dir,
            argv=sys.argv,
            config_path=args.config,
            eval_only=bool(args.eval_only),
            repo_root=Path(__file__).resolve().parents[1],
            update_latest=True,
        )

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
        freeze_summary = freeze_model_modules(model, freeze_modules)
        print(
            "[Train] Frozen "
            f"{freeze_summary['frozen_parameter_tensors']} parameters matching prefixes: {freeze_modules}; "
            f"matched={freeze_summary['matched_prefixes']}"
        )

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
            if is_vc_suda_enabled(config):
                images = batch["source_images"].to(device)
                depths = batch["source_depths"].to(device)
                padding_masks = batch.get("source_padding_masks", None)
                noise_masks = batch.get("source_noise_masks", None)
            else:
                images = batch["images"].to(device)
                depths = batch["depths"].to(device)
                padding_masks = batch.get("padding_masks", None)
                noise_masks = batch.get("noise_masks", None)
            if padding_masks is not None:
                padding_masks = padding_masks.to(device)
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

            depth_noise_cfg = getattr(config.data, "depth_noise", None)
            should_abort, reasons = should_abort_for_depth_sanity(
                report,
                depth_gaussian_std=getattr(depth_noise_cfg, "gaussian_std", 0.0),
                depth_speckle_std=getattr(depth_noise_cfg, "speckle_std", 0.0),
            )
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
