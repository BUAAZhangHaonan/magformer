#!/usr/bin/env python3
"""Static eval protocol checker for fixed MagFormer VC-SUDA eval contracts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_TEACHER_WEIGHTS = Path("output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth")


class ProtocolError(ValueError):
    """Raised when an eval invocation does not match the fixed protocol."""


@dataclass(frozen=True)
class DepthContract:
    clip_min: float
    clip_max: float
    norm: str
    per_sample_norm: bool


@dataclass(frozen=True)
class ProtocolContract:
    name: str
    base_config: Path
    dataset_root: Path
    ann: str
    split: str
    image_size: int
    max_images: int | None
    score_threshold: float
    mask_threshold: float
    iou_types: str
    inference_topk: int
    max_dets: int
    depth: DepthContract
    annotation_image_count: int
    weights: Path | None
    require_default_weights: bool


PROTOCOLS: dict[str, ProtocolContract] = {
    "original_first50_teacher": ProtocolContract(
        name="original_first50_teacher",
        base_config=Path("configs/finetune_1k_full_1024.yaml"),
        dataset_root=Path("magformer_datasets/20260318_1K_1566"),
        ann="annotations/instances_all.json",
        split="all",
        image_size=1024,
        max_images=50,
        score_threshold=0.05,
        mask_threshold=0.5,
        iou_types="bbox,segm",
        inference_topk=100,
        max_dets=100,
        depth=DepthContract(
            clip_min=1.0015300512313843,
            clip_max=2.095623016357422,
            norm="minmax",
            per_sample_norm=True,
        ),
        annotation_image_count=1566,
        weights=ORIGINAL_TEACHER_WEIGHTS,
        require_default_weights=True,
    ),
    "pseudo_real_target_unlabeled200": ProtocolContract(
        name="pseudo_real_target_unlabeled200",
        base_config=Path("configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml"),
        dataset_root=Path("magformer_datasets/pseudo_real_512"),
        ann="annotations/instances_target_unlabeled.json",
        split="train",
        image_size=1024,
        max_images=None,
        score_threshold=0.05,
        mask_threshold=0.5,
        iou_types="bbox,segm",
        inference_topk=200,
        max_dets=200,
        depth=DepthContract(
            clip_min=0.0,
            clip_max=2.095623016357422,
            norm="minmax",
            per_sample_norm=True,
        ),
        annotation_image_count=200,
        weights=None,
        require_default_weights=False,
    ),
}


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    full_path = _repo_path(path)
    try:
        return str(full_path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(full_path)


def _same_repo_path(raw: Path, expected: Path) -> bool:
    return _repo_path(raw).resolve() == _repo_path(expected).resolve()


def _same_dataset_root(raw: Path, expected: Path) -> bool:
    raw_resolved = _repo_path(raw).resolve()
    expected_resolved = _repo_path(expected).resolve()
    if raw_resolved == expected_resolved:
        return True
    expected_parts = expected.parts
    return len(raw_resolved.parts) >= len(expected_parts) and raw_resolved.parts[-len(expected_parts) :] == expected_parts


def _float_equal(left: Any, right: float) -> bool:
    try:
        return abs(float(left) - right) <= 1e-9
    except (TypeError, ValueError):
        return False


def _load_yaml(path: Path) -> dict[str, Any]:
    full_path = _repo_path(path)
    if not full_path.exists():
        raise ProtocolError(f"base_config does not exist: {_display_path(full_path)}")
    with full_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ProtocolError(f"base_config must be a YAML mapping: {_display_path(full_path)}")
    return loaded


def _load_annotation_image_count(dataset_root: Path, ann: str) -> int:
    ann_path = _repo_path(dataset_root) / ann
    if not ann_path.exists():
        raise ProtocolError(f"annotation does not exist: {_display_path(ann_path)}")
    with ann_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    images = payload.get("images")
    if not isinstance(images, list):
        raise ProtocolError(f"annotation images must be a list: {_display_path(ann_path)}")
    return len(images)


def _append_path_mismatch(errors: list[str], field: str, actual: Path, expected: Path) -> None:
    errors.append(f"{field} mismatch: expected {expected}, got {actual}")


def _append_value_mismatch(errors: list[str], field: str, actual: object, expected: object) -> None:
    errors.append(f"{field} mismatch: expected {expected!r}, got {actual!r}")


def _validate_depth(
    *,
    errors: list[str],
    protocol: ProtocolContract,
    depth_cfg: dict[str, Any],
) -> None:
    expected = protocol.depth
    checks = {
        "clip_min": _float_equal(depth_cfg.get("clip_min"), expected.clip_min),
        "clip_max": _float_equal(depth_cfg.get("clip_max"), expected.clip_max),
        "norm": depth_cfg.get("norm") == expected.norm,
        "per_sample_norm": bool(depth_cfg.get("per_sample_norm")) is expected.per_sample_norm,
    }
    if all(checks.values()):
        return
    errors.append(
        "depth config mismatch for protocol "
        f"{protocol.name}: expected clip {expected.clip_min}/{expected.clip_max}, "
        f"norm {expected.norm}, per_sample_norm {expected.per_sample_norm}; "
        f"got clip {depth_cfg.get('clip_min')}/{depth_cfg.get('clip_max')}, "
        f"norm {depth_cfg.get('norm')}, per_sample_norm {depth_cfg.get('per_sample_norm')}"
    )


def _check_weights(args: argparse.Namespace, protocol: ProtocolContract, errors: list[str]) -> dict[str, object]:
    weights = Path(args.weights)
    if not _repo_path(weights).exists():
        errors.append(f"weights path does not exist: {weights}")
    allow_nondefault = bool(args.allow_nondefault_weights)
    if protocol.require_default_weights and protocol.weights is not None and not allow_nondefault:
        if not _same_repo_path(weights, protocol.weights):
            errors.append(
                f"weights mismatch: {protocol.name} requires default Teacher weights {protocol.weights}; "
                "pass --allow-nondefault-weights only for an intentional non-Teacher original first50 check"
            )
    return {
        "path": str(weights),
        "exists": _repo_path(weights).exists(),
        "allow_nondefault": allow_nondefault,
        "default_required": protocol.require_default_weights,
        "default_path": str(protocol.weights) if protocol.weights is not None else None,
    }


def _validate_fixed_fields(args: argparse.Namespace, protocol: ProtocolContract, errors: list[str]) -> None:
    if not _same_repo_path(Path(args.base_config), protocol.base_config):
        _append_path_mismatch(errors, "base_config", Path(args.base_config), protocol.base_config)
    if not _same_dataset_root(Path(args.dataset_root), protocol.dataset_root):
        _append_path_mismatch(errors, "dataset_root", Path(args.dataset_root), protocol.dataset_root)
    if args.ann != protocol.ann:
        _append_value_mismatch(errors, "ann", args.ann, protocol.ann)
    if args.split != protocol.split:
        _append_value_mismatch(errors, "split", args.split, protocol.split)
    if args.image_size != protocol.image_size:
        _append_value_mismatch(errors, "image_size", args.image_size, protocol.image_size)
    if args.max_images != protocol.max_images:
        _append_value_mismatch(errors, "max_images", args.max_images, protocol.max_images)
    if not _float_equal(args.score_threshold, protocol.score_threshold):
        _append_value_mismatch(errors, "score_threshold", args.score_threshold, protocol.score_threshold)
    if not _float_equal(args.mask_threshold, protocol.mask_threshold):
        _append_value_mismatch(errors, "mask_threshold", args.mask_threshold, protocol.mask_threshold)
    if args.iou_types != protocol.iou_types:
        _append_value_mismatch(errors, "iou_types", args.iou_types, protocol.iou_types)
    if args.inference_topk != protocol.inference_topk:
        _append_value_mismatch(errors, "inference_topk", args.inference_topk, protocol.inference_topk)
    if args.max_dets != protocol.max_dets:
        _append_value_mismatch(errors, "max_dets", args.max_dets, protocol.max_dets)


def _validate_base_config(args: argparse.Namespace, protocol: ProtocolContract, errors: list[str]) -> dict[str, object]:
    cfg = _load_yaml(Path(args.base_config))
    data_cfg = cfg.get("data") or {}
    if not isinstance(data_cfg, dict):
        errors.append(f"base_config data must be a mapping for protocol {protocol.name}")
        data_cfg = {}
    depth_cfg = data_cfg.get("depth") or {}
    if not isinstance(depth_cfg, dict):
        errors.append(f"base_config data.depth must be a mapping for protocol {protocol.name}")
        depth_cfg = {}

    config_dataset_root = Path(str(data_cfg.get("dataset_root", "")))
    if not _same_dataset_root(config_dataset_root, protocol.dataset_root):
        _append_path_mismatch(errors, "base_config data.dataset_root", config_dataset_root, protocol.dataset_root)
    if int(data_cfg.get("image_size", -1)) != protocol.image_size:
        _append_value_mismatch(errors, "base_config data.image_size", data_cfg.get("image_size"), protocol.image_size)
    _validate_depth(errors=errors, protocol=protocol, depth_cfg=depth_cfg)

    return {
        "path": _display_path(Path(args.base_config)),
        "data_dataset_root": str(config_dataset_root),
        "data_image_size": data_cfg.get("image_size"),
        "depth": {
            "clip_min": depth_cfg.get("clip_min"),
            "clip_max": depth_cfg.get("clip_max"),
            "norm": depth_cfg.get("norm"),
            "per_sample_norm": depth_cfg.get("per_sample_norm"),
        },
    }


def run_check(args: argparse.Namespace) -> dict[str, object]:
    protocol = PROTOCOLS[args.protocol]
    errors: list[str] = []

    _validate_fixed_fields(args, protocol, errors)
    base_config_summary = _validate_base_config(args, protocol, errors)
    weights_summary = _check_weights(args, protocol, errors)
    image_count = _load_annotation_image_count(Path(args.dataset_root), args.ann)
    if image_count != protocol.annotation_image_count:
        _append_value_mismatch(errors, "annotation image_count", image_count, protocol.annotation_image_count)

    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise ProtocolError(f"protocol {protocol.name} failed static checks:\n{detail}")

    return {
        "protocol": protocol.name,
        "checks": {"status": "pass"},
        "cli": {
            "base_config": args.base_config,
            "dataset_root": args.dataset_root,
            "ann": args.ann,
            "split": args.split,
            "image_size": args.image_size,
            "max_images": args.max_images,
            "score_threshold": args.score_threshold,
            "mask_threshold": args.mask_threshold,
            "iou_types": args.iou_types,
            "inference_topk": args.inference_topk,
            "max_dets": args.max_dets,
        },
        "base_config": base_config_summary,
        "weights": weights_summary,
        "annotation": {
            "path": str(Path(args.dataset_root) / args.ann),
            "image_count": image_count,
            "expected_image_count": protocol.annotation_image_count,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statically verify a fixed MagFormer eval protocol without training or model eval."
    )
    parser.add_argument("protocol", choices=sorted(PROTOCOLS))
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--mask-threshold", type=float, required=True)
    parser.add_argument("--iou-types", required=True)
    parser.add_argument("--inference-topk", type=int, required=True)
    parser.add_argument("--max-dets", type=int, required=True)
    parser.add_argument("--allow-nondefault-weights", action="store_true")
    parser.add_argument("--summary-json", default=None, help="Optional path to write the JSON summary.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_check(args)
    except ProtocolError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output = json.dumps(summary, indent=2, sort_keys=True)
    print(output)
    if args.summary_json:
        summary_path = _repo_path(Path(args.summary_json))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
