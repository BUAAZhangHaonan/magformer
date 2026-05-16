#!/usr/bin/env python3
"""Fail-fast VC-SUDA stage preflight checks.

This script is safe to run before formal training. With the default options it
validates static config semantics and reads one CPU batch. It does not build a
model, load a checkpoint into a model, start training, or write outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from magformer.config import load_config  # noqa: E402
from tools import train as train_tool  # noqa: E402


LABEL_KEYS = ("labels", "masks", "boxes", "annotations")


class PreflightError(RuntimeError):
    """Raised when a VC-SUDA preflight gate fails."""


@dataclass
class PreflightResult:
    config_path: Path
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def _fail(message: str) -> None:
    raise PreflightError(message)


def _resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _ann_path(dataset_root: str, ann_file: str) -> Path:
    ann = Path(ann_file).expanduser()
    if ann.is_absolute():
        return ann.resolve()
    return (PROJECT_ROOT / dataset_root / ann).resolve()


def _dataset_root_path(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


def _load_coco_ann(path: Path) -> dict[str, Any]:
    if not path.exists():
        _fail(f"Annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "images" not in payload:
        _fail(f"Annotation file has no images list: {path}")
    return payload


def _image_path(root: Path, split: str, file_name: str) -> Path:
    image_dir = root / "images" / split
    direct = image_dir / file_name
    if direct.exists():
        return direct
    basename = image_dir / Path(file_name).name
    if basename.exists():
        return basename
    _fail(f"RGB file not found for {file_name}: tried {direct} and {basename}")


def _depth_path(root: Path, split: str, file_name: str) -> Path:
    stem = Path(file_name).stem
    candidates = [
        root / "depth" / "depth_npy" / split / f"{stem}.npy",
        root / "depth" / "depth_npy" / split / f"{stem}.npz",
        root / "depth" / split / f"{stem}.npy",
        root / "depth" / split / f"{stem}.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    _fail(f"Depth file not found for {file_name}: tried {[str(path) for path in candidates]}")


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _load_depth_array(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if isinstance(array, np.lib.npyio.NpzFile):
        with array:
            first_key = sorted(array.files)[0]
            return np.asarray(array[first_key])
    return np.asarray(array)


def _sample_file_evidence(root: Path, split: str, images: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    samples = []
    for image in images[:max_samples]:
        file_name = str(image.get("file_name", ""))
        if not file_name:
            _fail(f"Image record is missing file_name in {root} split={split}")
        rgb_path = _image_path(root, split, file_name)
        depth_path = _depth_path(root, split, file_name)
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            _fail(f"Failed to read RGB file: {rgb_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth = _load_depth_array(depth_path)
        samples.append(
            {
                "image_id": int(image["id"]),
                "file_name": file_name,
                "rgb": str(rgb_path),
                "depth": str(depth_path),
                "rgb_exists": rgb_path.exists(),
                "depth_exists": depth_path.exists(),
                "rgb_stats": _array_stats(rgb),
                "depth_stats": _array_stats(depth),
            }
        )
    return samples


def _split_evidence(root: Path, ann: Path, split: str, max_samples: int) -> dict[str, Any]:
    payload = _load_coco_ann(ann)
    images = list(payload.get("images", []))
    return {
        "root": str(root),
        "ann": str(ann),
        "split": split,
        "count": len(images),
        "samples": _sample_file_evidence(root, split, images, max_samples),
    }


def collect_source_target_evidence(cfg: Any, *, max_samples: int = 3) -> dict[str, Any]:
    if max_samples < 1:
        _fail("max_samples must be >= 1")
    data_cfg = cfg.data
    vc = cfg.vc_suda
    target_root = _dataset_root_path(data_cfg.dataset_root)
    source_root = _dataset_root_path(getattr(vc, "source_root", None) or data_cfg.dataset_root)
    train_split = getattr(data_cfg, "train_split", "train")
    source_split = getattr(vc, "source_split", train_split)
    target_labeled_split = getattr(vc, "target_labeled_split", train_split)
    target_unlabeled_split = getattr(vc, "target_unlabeled_split", train_split)
    val_split = getattr(data_cfg, "val_split", "val")

    source = _split_evidence(
        source_root,
        _ann_path(str(source_root), vc.source_ann),
        source_split,
        max_samples,
    )
    target_labeled = _split_evidence(
        target_root,
        _ann_path(str(target_root), vc.target_labeled_ann),
        target_labeled_split,
        max_samples,
    )
    target_unlabeled = _split_evidence(
        target_root,
        _ann_path(str(target_root), vc.target_unlabeled_ann),
        target_unlabeled_split,
        max_samples,
    )
    val = _split_evidence(target_root, _ann_path(str(target_root), data_cfg.val_ann), val_split, max_samples)
    return {
        "source": source,
        "target_labeled": target_labeled,
        "target_unlabeled": target_unlabeled,
        "val": val,
        "dataset_len": max(source["count"], target_unlabeled["count"]),
        "max_samples": max_samples,
    }


def _load_config_or_fail(config_path: Path):
    try:
        return load_config(str(config_path))
    except Exception as exc:
        raise PreflightError(str(exc)) from exc


def _check_static_config(cfg: Any, result: PreflightResult, require_stage: str) -> None:
    vc = cfg.vc_suda
    if not bool(vc.enabled):
        _fail("vc_suda.enabled must be true for Stage C preflight.")

    stage = str(vc.stage).upper()
    if stage != require_stage:
        _fail(f"vc_suda.stage must be {require_stage}, got {stage}.")
    result.checks.append("stage")

    target_unlabeled_ann = getattr(vc, "target_unlabeled_ann", None)
    if not target_unlabeled_ann:
        _fail("vc_suda.target_unlabeled_ann is required for Stage C.")

    target_unlabeled_split = getattr(vc, "target_unlabeled_split", cfg.data.train_split)
    if target_unlabeled_split != cfg.data.train_split:
        _fail(
            "target_unlabeled split must use the training split; "
            f"got {target_unlabeled_split!r}, train_split={cfg.data.train_split!r}."
        )
    if target_unlabeled_split == cfg.data.val_split:
        _fail("target_unlabeled split must not reuse data.val_split.")
    result.details["target_unlabeled_split"] = target_unlabeled_split
    result.checks.append("unlabeled_split")

    eval_period = getattr(cfg.runtime, "eval_period", None)
    if isinstance(eval_period, bool) or not isinstance(eval_period, int) or eval_period <= 0:
        _fail("runtime.eval_period must be a positive integer for Stage C quick eval.")
    result.details["eval_period"] = eval_period
    result.checks.append("eval_period")

    eval_iou_types = list(cfg.runtime.eval_iou_types or [])
    if "bbox" not in set(eval_iou_types):
        _fail("runtime.eval_iou_types must include bbox for Stage C quick eval.")
    result.details["eval_iou_types"] = eval_iou_types
    result.checks.append("eval_iou_types")

    eval_max_images = getattr(cfg.runtime, "eval_max_images", None)
    if (
        isinstance(eval_max_images, bool)
        or not isinstance(eval_max_images, int)
        or not math.isfinite(float(eval_max_images))
        or eval_max_images <= 0
        or eval_max_images >= 99999
    ):
        _fail("runtime.eval_max_images must be a finite positive subset size for Stage C quick eval.")
    result.details["eval_max_images"] = eval_max_images
    result.checks.append("eval_max_images")

    eval_batch_size = getattr(cfg.runtime, "eval_batch_size", None)
    if isinstance(eval_batch_size, bool) or not isinstance(eval_batch_size, int) or eval_batch_size <= 1:
        _fail("runtime.eval_batch_size must be an integer > 1 for Stage C quick eval.")
    result.details["eval_batch_size"] = eval_batch_size
    result.checks.append("eval_batch_size")

    eval_saves_best = getattr(cfg.runtime, "eval_saves_best", None)
    if eval_saves_best is not False:
        _fail("runtime.eval_saves_best must be false for Stage C quick eval.")
    result.details["eval_saves_best"] = eval_saves_best
    result.checks.append("eval_saves_best")

    unsupervised_weight = float(vc.unsupervised_weight)
    if not math.isfinite(unsupervised_weight) or unsupervised_weight < 0.0:
        _fail("vc_suda.unsupervised_weight must be >= 0 for Stage C.")
    result.details["unsupervised_weight"] = unsupervised_weight
    result.checks.append("unsupervised_weight")

    ema_cfg = getattr(vc, "ema_teacher", None)
    if ema_cfg is None or not bool(getattr(ema_cfg, "enabled", False)):
        _fail("vc_suda.ema_teacher.enabled must be true for Stage C.")
    result.checks.append("ema_teacher")

    if bool(getattr(cfg.runtime, "ema_enabled", False)):
        _fail(
            "runtime.ema_enabled must be false for Stage C; "
            "use vc_suda.ema_teacher.enabled for pseudo-label teacher EMA only."
        )
    result.checks.append("runtime_no_generic_ema")

    if cfg.data.depth.norm != "minmax" or bool(cfg.data.depth.per_sample_norm) is not True:
        _fail("Stage C must preserve depth norm: data.depth.norm=minmax and per_sample_norm=true.")
    result.checks.append("depth_norm")


def _coco_image_keys(path: Path) -> set[str]:
    if not path.exists():
        _fail(f"Annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    keys = set()
    for image in data.get("images", []):
        file_name = str(image.get("file_name", "")).replace("\\", "/").strip()
        if file_name:
            keys.add(file_name)
            keys.add(Path(file_name).name)
    return keys


def _check_annotation_splits(cfg: Any, result: PreflightResult) -> None:
    dataset_root = cfg.data.dataset_root
    target_path = _ann_path(dataset_root, cfg.vc_suda.target_unlabeled_ann)
    val_path = _ann_path(dataset_root, cfg.data.val_ann)

    target_key = str(Path(cfg.vc_suda.target_unlabeled_ann)).replace("\\", "/").lstrip("./")
    val_key = str(Path(cfg.data.val_ann)).replace("\\", "/").lstrip("./")
    if target_key == val_key:
        _fail("vc_suda.target_unlabeled_ann must not match data.val_ann.")

    target_images = _coco_image_keys(target_path)
    val_images = _coco_image_keys(val_path)
    overlap = sorted(target_images & val_images)
    if overlap:
        _fail(
            "target_unlabeled and val annotations overlap by image file; "
            f"first overlaps: {overlap[:5]}"
        )

    result.details["target_unlabeled_images"] = len(target_images)
    result.details["val_images"] = len(val_images)
    result.checks.append("target_unlabeled_val_overlap")


def _check_checkpoint_semantics(cfg: Any, result: PreflightResult, require_finetune_exists: bool) -> None:
    resume = getattr(cfg.runtime, "resume", None)
    finetune_weights = getattr(cfg.model, "finetune_weights", None)

    if resume is not None:
        if finetune_weights:
            _fail("model.finetune_weights must be null when runtime.resume is set for true resume.")

        normalized = str(resume).replace("\\", "/")
        checkpoint_name = Path(normalized).name
        resume_checkpoint_role = None
        if "stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499" in normalized:
            if checkpoint_name != "checkpoint_iter_0000499.pth":
                _fail(
                    "runtime.resume must use R12 checkpoint_iter_0000499.pth "
                    "for the plus25 true-resume continuation."
                )
            resume_checkpoint_role = "r12_ckpt499_true_resume"
        else:
            _fail(
                "runtime.resume must identify the R12 ckpt499 full training checkpoint "
                "for true-resume Stage C continuation."
            )

        resume_path = _resolve_project_path(resume)
        if resume_path is None:
            _fail("runtime.resume could not be resolved.")
        if not resume_path.exists():
            if require_finetune_exists:
                _fail(f"runtime.resume does not exist: {resume_path}")
            result.warnings.append(
                "runtime.resume does not exist yet; allowed because checkpoint existence check is disabled."
            )
            result.details["resume"] = str(resume_path)
            result.details["resume_checkpoint_role"] = resume_checkpoint_role
            result.checks.append("checkpoint_semantics")
            return

        try:
            from torch._subclasses.fake_tensor import FakeTensorMode

            with FakeTensorMode():
                checkpoint = torch.load(
                    resume_path,
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
        except Exception as exc:
            raise PreflightError(f"Failed to read runtime.resume checkpoint: {resume_path}: {exc}") from exc
        if not isinstance(checkpoint, dict):
            _fail(f"runtime.resume checkpoint must be a dict: {resume_path}")

        required_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "lr_scheduler_state_dict",
            "scaler_state_dict",
            "ema_teacher_state_dict",
            "curriculum_state_dict",
        }
        missing = sorted(required_keys - set(checkpoint))
        if missing:
            _fail(
                "runtime.resume checkpoint is missing full training state keys: "
                + ", ".join(missing)
            )

        resume_iter = checkpoint.get("iter")
        if isinstance(resume_iter, bool) or not isinstance(resume_iter, int):
            _fail("runtime.resume checkpoint must contain integer iter.")
        if resume_iter != 499:
            _fail(f"runtime.resume R12 checkpoint must have iter=499, got {resume_iter}.")

        max_iter = getattr(cfg.solver, "max_iter", None)
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            _fail("solver.max_iter must be an integer for true resume.")
        if max_iter <= resume_iter:
            _fail(
                f"solver.max_iter must be greater than resume iter {resume_iter}, got {max_iter}."
            )
        if max_iter != 750:
            _fail(
                "R12 ckpt499 plus25 true resume must set solver.max_iter=750 "
                f"to reach the ckpt749/final gate, got {max_iter}."
            )

        result.details["resume"] = str(resume_path)
        result.details["resume_checkpoint_role"] = resume_checkpoint_role
        result.details["resume_iter"] = resume_iter
        result.details["max_iter"] = max_iter
        result.checks.append("checkpoint_semantics")
        return

    if not finetune_weights:
        _fail("model.finetune_weights must point to the completed Stage B final model checkpoint.")

    normalized = str(finetune_weights).replace("\\", "/")
    checkpoint_name = Path(normalized).name
    finetune_checkpoint_role = None
    if "stage_b_1024_teacher8499" in normalized:
        if checkpoint_name != "checkpoint_iter_0008999.pth":
            _fail(
                "model.finetune_weights must use Stage B checkpoint_iter_0008999.pth, "
                "not model_final.pth, runtime.resume, or another iter resume state."
            )
        finetune_checkpoint_role = "stage_b_final"
    elif "stage_c_r7_a10_lsj10_1024_teacher8499" in normalized:
        if checkpoint_name != "checkpoint_iter_0001999.pth":
            _fail(
                "model.finetune_weights must use R7 checkpoint_iter_0001999.pth "
                "for low-LR continuation."
            )
        finetune_checkpoint_role = "r7_ckpt1999_continuation"
    elif "stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499" in normalized:
        if checkpoint_name != "checkpoint_iter_0000999.pth":
            _fail(
                "model.finetune_weights must use R8B checkpoint_iter_0000999.pth "
                "for R10 no-depth-noise continuation."
            )
        finetune_checkpoint_role = "r8b_ckpt999_continuation"
    elif "stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499" in normalized:
        if checkpoint_name != "checkpoint_iter_0000499.pth":
            _fail(
                "model.finetune_weights must use R12 checkpoint_iter_0000499.pth "
                "for R18 pseudo-unmatched-negative continuation."
            )
        finetune_checkpoint_role = "r12_ckpt499_continuation"
    else:
        _fail(
            "model.finetune_weights must identify the Stage B teacher8499 final checkpoint, "
            "the R7 ckpt1999 continuation path, the R8B ckpt999 continuation path, "
            "or the R12 ckpt499 continuation path."
        )

    finetune_path = _resolve_project_path(finetune_weights)
    if finetune_path is None:
        _fail("model.finetune_weights could not be resolved.")
    if not finetune_path.exists():
        if require_finetune_exists:
            _fail(f"model.finetune_weights does not exist: {finetune_path}")
        result.warnings.append(
            "model.finetune_weights does not exist yet; allowed because Stage B final checkpoint is pending."
        )

    result.details["finetune_weights"] = str(finetune_path)
    result.details["finetune_checkpoint_role"] = finetune_checkpoint_role
    result.checks.append("checkpoint_semantics")

def _ensure_no_label_fields(mapping: dict[str, Any], context: str) -> None:
    leaked = [key for key in LABEL_KEYS if key in mapping]
    if leaked:
        _fail(f"{context} must not contain label fields: {leaked}")


def _depth_stats(depths: torch.Tensor) -> dict[str, float | int | list[int]]:
    depths = depths.detach().cpu().float()
    return {
        "shape": list(depths.shape),
        "min": float(depths.min().item()),
        "max": float(depths.max().item()),
        "std": float(depths.std().item()),
        "unique": int(torch.unique(depths).numel()),
    }


def _check_nonconstant_depth(depths: torch.Tensor, context: str) -> dict[str, Any]:
    stats = _depth_stats(depths)
    if stats["unique"] <= 1 or stats["std"] <= 1e-4:
        _fail(f"{context} depth is constant or near-constant: {stats}")
    return stats


def _check_one_unlabeled_batch(cfg: Any, result: PreflightResult) -> None:
    train_dataset, val_dataset = train_tool.build_datasets(cfg)
    if getattr(train_dataset, "target_unlabeled", None) is None:
        _fail("Stage C train dataset did not build target_unlabeled.")

    sample = train_dataset[0]
    for view_name in ("target_weak", "target_strong"):
        if view_name not in sample:
            _fail(f"Stage C sample is missing {view_name}.")
        _ensure_no_label_fields(sample[view_name], view_name)

    train_loader, _ = train_tool.build_data_loaders(
        cfg,
        train_dataset,
        val_dataset,
        batch_size=1,
        num_workers=0,
        is_distributed=False,
    )
    batch = next(iter(train_loader))
    for forbidden in LABEL_KEYS:
        for prefix in ("target_weak", "target_strong"):
            key = f"{prefix}_{forbidden}"
            if key in batch:
                _fail(f"Unlabeled batch must not contain {key}.")

    result.details["target_weak_depth"] = _check_nonconstant_depth(batch["target_weak_depths"], "target_weak")
    result.details["target_strong_depth"] = _check_nonconstant_depth(batch["target_strong_depths"], "target_strong")
    result.checks.append("unlabeled_batch_strip")
    result.checks.append("depth_nonconstant")


def run_preflight(
    config_path: str | Path,
    *,
    check_batch: bool = True,
    require_stage: str = "C",
    require_finetune_exists: bool = True,
    collect_data_evidence: bool = False,
    evidence_max_samples: int = 3,
) -> PreflightResult:
    path = Path(config_path)
    result = PreflightResult(config_path=path)
    cfg = _load_config_or_fail(path)

    _check_static_config(cfg, result, require_stage=require_stage)
    _check_annotation_splits(cfg, result)
    _check_checkpoint_semantics(cfg, result, require_finetune_exists=require_finetune_exists)
    if collect_data_evidence:
        result.details["data_evidence"] = collect_source_target_evidence(
            cfg,
            max_samples=evidence_max_samples,
        )
        result.checks.append("source_target_evidence")
    if check_batch:
        _check_one_unlabeled_batch(cfg, result)
    return result


def _format_lines(result: PreflightResult) -> Iterable[str]:
    yield f"PASS config={result.config_path}"
    yield "checks=" + ",".join(result.checks)
    for warning in result.warnings:
        yield "WARNING " + warning
    yield "details=" + json.dumps(result.details, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vc_suda_stage_c_1024_teacher8499.yaml")
    parser.add_argument("--stage", default="C")
    parser.add_argument("--skip-batch", action="store_true", help="Only run static config and annotation checks.")
    parser.add_argument(
        "--allow-missing-finetune",
        action="store_true",
        help="Allow the Stage B final checkpoint placeholder to be absent while Stage B is still running.",
    )
    parser.add_argument(
        "--emit-data-evidence",
        action="store_true",
        help="Record source/target root, annotation, count, and sampled RGB/depth stats.",
    )
    parser.add_argument("--evidence-max-samples", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        result = run_preflight(
            args.config,
            check_batch=not args.skip_batch,
            require_stage=args.stage.upper(),
            require_finetune_exists=not args.allow_missing_finetune,
            collect_data_evidence=args.emit_data_evidence,
            evidence_max_samples=args.evidence_max_samples,
        )
    except PreflightError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1

    for line in _format_lines(result):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
