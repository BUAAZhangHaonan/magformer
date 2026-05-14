#!/usr/bin/env python3
"""
MAGFormer Evaluation Script (Pure PyTorch)

Weight selection priority is explicit: --weights > config.model.weights > error.

Usage:
    python tools/evaluate.py --config-file configs/magformer.yaml --dataset-root /path/to/eccd
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from magformer.config import load_config, setup_device, set_seed
from magformer.data import CocoRgbdDataset
from magformer.data.transforms import RGBDTransform
from magformer.data.collate import collate_fn
from magformer.data.eval_subset import build_global_eval_subset
from magformer.engine.eval_runtime import run_inference_evaluation
from magformer.models import build_model
from magformer.engine.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MAGFormer model")
    parser.add_argument("--config-file", required=True, help="Path to config yaml")
    parser.add_argument("--dataset-root", required=False, help="Override dataset root")
    parser.add_argument("--weights", required=False, help="Checkpoint path")
    parser.add_argument("--output", default="output/eval", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    return parser.parse_args()


def build_val_loader(config, dataset_root_override=None, num_workers=4, batch_size=1):
    data_cfg = config.data
    dataset_root = dataset_root_override or data_cfg.dataset_root
    val_split = getattr(data_cfg, "val_split", "val")

    dataset = CocoRgbdDataset(
        dataset_root=dataset_root,
        ann_file=data_cfg.val_ann,
        split=val_split,
        transform=None,
        is_train=False,
    )

    dataset.transform = RGBDTransform(
        image_size=data_cfg.image_size,
        min_scale=data_cfg.min_scale,
        max_scale=data_cfg.max_scale,
        random_flip="none",
        rgb_brightness=0.0,
        rgb_contrast=0.0,
        rgb_saturation=0.0,
        rgb_hue=0.0,
        depth_scale=data_cfg.depth.scale,
        depth_shift=data_cfg.depth.shift,
        depth_clip_min=data_cfg.depth.clip_min,
        depth_clip_max=data_cfg.depth.clip_max,
        depth_norm=data_cfg.depth.norm,
        depth_per_sample_norm=getattr(data_cfg.depth, "per_sample_norm", True),
        is_train=False,
    )
    dataset = build_global_eval_subset(
        dataset,
        getattr(getattr(config, "runtime", None), "eval_max_images", None),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return dataset, loader


def main() -> None:
    args = parse_args()

    overrides = {}
    if args.dataset_root is not None:
        overrides.setdefault("data", {})["dataset_root"] = args.dataset_root
    if args.weights is not None:
        overrides.setdefault("model", {})["weights"] = args.weights
    overrides.setdefault("runtime", {})["output_dir"] = args.output

    config = load_config(args.config_file, overrides=overrides)

    device = setup_device(config.runtime)
    set_seed(config.runtime.seed)

    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, loader = build_val_loader(
        config,
        dataset_root_override=args.dataset_root,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    # Explicit priority: CLI --weights wins, then config.model.weights, then fail loudly.
    effective_weights = args.weights or getattr(config.model, "weights", None)
    if not effective_weights:
        raise ValueError(
            "Evaluation requires weights. Priority is --weights > config.model.weights > explicit error."
        )

    model = build_model(config)
    load_checkpoint(effective_weights, model, strict=True)
    model = model.to(device)
    model.eval()

    result = run_inference_evaluation(
        model,
        loader,
        coco_gt=dataset.coco,
        device=device,
        output_dir=output_dir,
        amp_enabled=False,
        category_ids=list(getattr(dataset, "category_ids", [])) or None,
        iou_types=getattr(config.runtime, "eval_iou_types", None),
        max_images=getattr(config.runtime, "eval_max_images", None),
        fail_on_empty=True,
    )

    print(f"[Eval] Results saved to {result.coco_results_path}")
    print(result.coco_metrics)


if __name__ == "__main__":
    main()
