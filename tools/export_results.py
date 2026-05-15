#!/usr/bin/env python3
"""
Export COCO Results

Runs inference on validation set and exports coco_instances_results.json.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from magformer.config import load_config, setup_device
from magformer.data import CocoRgbdDataset
from magformer.data.transforms import RGBDTransform
from magformer.data.collate import collate_fn
from magformer.engine.eval_runtime import run_inference_evaluation
from magformer.models import build_model
from magformer.engine.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export COCO results")
    parser.add_argument("--config-file", required=True, help="Path to config yaml")
    parser.add_argument("--dataset-root", required=False, help="Override dataset root")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--output", default="output/eval", help="Output directory")
    parser.add_argument("--inference-topk", type=int, default=None, help="Override runtime.eval_inference_topk")
    parser.add_argument("--max-dets", type=int, default=None, help="Override runtime.eval_max_dets")
    return parser.parse_args()


def _require_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def build_val_dataset(config, dataset_root: str):
    dataset = CocoRgbdDataset(
        dataset_root=dataset_root,
        ann_file=config.data.val_ann,
        split="val",
        transform=None,
        is_train=False,
    )
    dataset.transform = RGBDTransform(
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
    return dataset


def main() -> None:
    args = parse_args()

    overrides = {"runtime": {"output_dir": args.output}}
    if args.dataset_root is not None:
        overrides.setdefault("data", {})["dataset_root"] = args.dataset_root
    if args.inference_topk is not None:
        overrides["runtime"]["eval_inference_topk"] = _require_positive_int(
            args.inference_topk,
            "--inference-topk",
        )
    if args.max_dets is not None:
        overrides["runtime"]["eval_max_dets"] = _require_positive_int(args.max_dets, "--max-dets")

    config = load_config(args.config_file, overrides=overrides)

    device = setup_device(config.runtime)

    dataset_root = args.dataset_root or config.data.dataset_root
    dataset = build_val_dataset(config, dataset_root=dataset_root)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_model(config)
    load_checkpoint(args.weights, model, strict=True)
    model = model.to(device)
    model.eval()

    result = run_inference_evaluation(
        model,
        loader,
        coco_gt=dataset.coco,
        device=device,
        output_dir=args.output,
        amp_enabled=False,
        score_threshold=0.0,
        category_ids=list(getattr(dataset, "category_ids", [])) or None,
        fail_on_empty=True,
        inference_topk=getattr(config.runtime, "eval_inference_topk", 100),
        max_dets=getattr(config.runtime, "eval_max_dets", 100),
    )

    print(f"[Export] Saved results to {result.coco_results_path}")


if __name__ == "__main__":
    main()
