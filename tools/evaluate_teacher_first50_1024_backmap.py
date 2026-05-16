#!/usr/bin/env python3
"""Guarded Teacher 8499 original 1.5K first50 1024 backmap evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BASE_CONFIG = Path("configs/finetune_1k_full_1024.yaml")
EXPECTED_DATASET_ROOT = Path("magformer_datasets/20260318_1K_1566")
EXPECTED_ANN = "annotations/instances_all.json"
EXPECTED_SPLIT = "all"
EXPECTED_IMAGE_SIZE = 1024
EXPECTED_MAX_IMAGES = 50
EXPECTED_IOU_TYPES = "bbox,segm"
EXPECTED_DEPTH_CLIP_MIN = 1.0015300512313843
EXPECTED_DEPTH_CLIP_MAX = 2.095623016357422


class ProtocolError(ValueError):
    """Raised when CLI inputs or base config do not match the fixed protocol."""


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _same_path(left: Path, right: Path) -> bool:
    return _repo_path(left).resolve() == _repo_path(right).resolve()


def _same_dataset_root(raw: Path, expected: Path) -> bool:
    resolved = _repo_path(raw).resolve()
    expected_resolved = _repo_path(expected).resolve()
    if resolved == expected_resolved:
        return True
    return resolved.parts[-len(expected.parts) :] == expected.parts


def _load_yaml(path: Path) -> dict[str, Any]:
    full_path = _repo_path(path)
    if not full_path.exists():
        raise ProtocolError(f"base config does not exist: {_display_path(full_path)}")
    with full_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ProtocolError(f"base config must be a YAML mapping: {_display_path(full_path)}")
    return loaded


def _float_equal(left: Any, right: float) -> bool:
    try:
        return abs(float(left) - right) <= 1e-9
    except (TypeError, ValueError):
        return False


def validate_original_first50_protocol(
    *,
    base_config: Path,
    dataset_root: Path,
    ann: str,
    split: str,
    image_size: int,
    max_images: int,
) -> dict[str, Any]:
    cfg = _load_yaml(base_config)
    data_cfg = cfg.get("data") or {}
    depth_cfg = data_cfg.get("depth") or {}
    errors: list[str] = []

    if not _same_path(base_config, EXPECTED_BASE_CONFIG):
        errors.append(
            "base config mismatch: original 1.5K first50 Teacher requires "
            f"{EXPECTED_BASE_CONFIG}, got {base_config}"
        )

    config_dataset_root = Path(str(data_cfg.get("dataset_root", "")))
    if not _same_dataset_root(config_dataset_root, EXPECTED_DATASET_ROOT):
        marker = " (pseudo_real)" if "pseudo_real" in str(config_dataset_root) else ""
        errors.append(
            "config dataset_root mismatch: expected original "
            f"{EXPECTED_DATASET_ROOT}, got {config_dataset_root}{marker}"
        )
    if data_cfg.get("val_ann") != EXPECTED_ANN:
        errors.append(f"config val_ann mismatch: expected {EXPECTED_ANN}, got {data_cfg.get('val_ann')}")
    if data_cfg.get("val_split") != EXPECTED_SPLIT:
        errors.append(f"config val_split mismatch: expected {EXPECTED_SPLIT}, got {data_cfg.get('val_split')}")
    if int(data_cfg.get("image_size", -1)) != EXPECTED_IMAGE_SIZE:
        errors.append(f"config image_size mismatch: expected {EXPECTED_IMAGE_SIZE}, got {data_cfg.get('image_size')}")

    clip_min = depth_cfg.get("clip_min")
    clip_max = depth_cfg.get("clip_max")
    if not (_float_equal(clip_min, EXPECTED_DEPTH_CLIP_MIN) and _float_equal(clip_max, EXPECTED_DEPTH_CLIP_MAX)):
        errors.append(
            "depth clip mismatch: original 1.5K first50 Teacher requires "
            f"{EXPECTED_DEPTH_CLIP_MIN}/{EXPECTED_DEPTH_CLIP_MAX}, got {clip_min}/{clip_max}"
        )

    if not _same_dataset_root(dataset_root, EXPECTED_DATASET_ROOT):
        marker = " (pseudo_real)" if "pseudo_real" in str(dataset_root) else ""
        errors.append(
            "CLI dataset_root mismatch: expected original "
            f"{EXPECTED_DATASET_ROOT}, got {dataset_root}{marker}"
        )
    if ann != EXPECTED_ANN:
        errors.append(f"CLI ann mismatch: expected {EXPECTED_ANN}, got {ann}")
    if split != EXPECTED_SPLIT:
        errors.append(f"CLI split mismatch: expected {EXPECTED_SPLIT}, got {split}")
    if int(image_size) != EXPECTED_IMAGE_SIZE:
        errors.append(f"CLI image_size mismatch: expected {EXPECTED_IMAGE_SIZE}, got {image_size}")
    if int(max_images) != EXPECTED_MAX_IMAGES:
        errors.append(f"CLI max_images mismatch: expected {EXPECTED_MAX_IMAGES}, got {max_images}")

    evidence = {
        "base_config": _display_path(_repo_path(base_config)),
        "config_dataset_root": str(config_dataset_root),
        "cli_dataset_root": str(dataset_root),
        "ann": ann,
        "split": split,
        "image_size": int(image_size),
        "max_images": int(max_images),
        "depth_clip_min": float(clip_min),
        "depth_clip_max": float(clip_max),
    }
    if errors:
        raise ProtocolError(
            "Original 1.5K first50 Teacher protocol mismatch:\n"
            + "\n".join(f"- {error}" for error in errors)
            + "\nDo not mix original first50 with pseudo_real or Stage B base configs."
        )
    return evidence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Teacher 8499 on the original 1.5K all split first50 with fixed "
            "1024 backmap settings and explicit depth/data protocol guards."
        )
    )
    parser.add_argument("--base-config", default=str(EXPECTED_BASE_CONFIG))
    parser.add_argument("--dataset-root", default=str(EXPECTED_DATASET_ROOT))
    parser.add_argument("--ann", default=EXPECTED_ANN)
    parser.add_argument("--split", default=EXPECTED_SPLIT)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=EXPECTED_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--inference-topk", type=int, default=100)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=EXPECTED_MAX_IMAGES)
    parser.add_argument("--iou-types", default=EXPECTED_IOU_TYPES)
    parser.add_argument("--dump-inference-stats", default=None)
    return parser.parse_args(argv)


def build_eval_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "evaluate_1024_backmap.py"),
        "--iou-types",
        args.iou_types,
        "--base-config",
        args.base_config,
        "--dataset-root",
        args.dataset_root,
        "--ann",
        args.ann,
        "--split",
        args.split,
        "--weights",
        args.weights,
        "--output-dir",
        args.output_dir,
        "--image-size",
        str(args.image_size),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--score-threshold",
        str(args.score_threshold),
        "--mask-threshold",
        str(args.mask_threshold),
        "--inference-topk",
        str(args.inference_topk),
        "--max-dets",
        str(args.max_dets),
        "--max-images",
        str(args.max_images),
        "--force-pytorch-msda",
    ]
    if args.dump_inference_stats:
        command.extend(["--dump-inference-stats", args.dump_inference_stats])
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.iou_types != EXPECTED_IOU_TYPES:
        raise ProtocolError(f"--iou-types must be {EXPECTED_IOU_TYPES!r} for this protocol, got {args.iou_types!r}")

    evidence = validate_original_first50_protocol(
        base_config=Path(args.base_config),
        dataset_root=Path(args.dataset_root),
        ann=args.ann,
        split=args.split,
        image_size=args.image_size,
        max_images=args.max_images,
    )
    print("[Protocol] original_first50_teacher=" + json.dumps(evidence, sort_keys=True), flush=True)

    env = os.environ.copy()
    env["MAGFORMER_MS_DEFORM_ATTN_BACKEND"] = "pytorch"
    command = build_eval_command(args)
    print("[Protocol] command=" + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProtocolError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
