#!/usr/bin/env python3
"""
Multi-resolution evaluation for MagFormer.

Evaluates a single checkpoint at multiple input resolutions.
Reuses the same model (resolution-agnostic) and rebuilds the dataloader
with a different image_size for each resolution.

Usage:
    python tools/eval_multi_res.py \
        --config-file configs/finetune_1k_full_1024_v14.yaml \
        --checkpoint output/.../checkpoint_best.pth \
        --resolutions 512,768,1024,1280

    # Quick test with max-images
    python tools/eval_multi_res.py \
        --config-file configs/finetune_1k_full_1024_v14.yaml \
        --checkpoint output/.../checkpoint_best.pth \
        --resolutions 512,1024 \
        --max-images 50
"""

import argparse
import os
import sys
import time
import gc
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader

from magformer.config import load_config, setup_device, set_seed
from magformer.data import CocoRgbdDataset
from magformer.data.transforms import RGBDTransform
from magformer.data.collate import collate_fn
from magformer.engine.eval_runtime import run_inference_evaluation
from magformer.models import build_model
from magformer.engine.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MagFormer at multiple input resolutions"
    )
    parser.add_argument("--config-file", required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument(
        "--resolutions",
        default="512,768,1024,1280",
        help="Comma-separated resolutions (default: 512,768,1024,1280)",
    )
    parser.add_argument("--dataset-root", default=None, help="Override dataset root")
    parser.add_argument("--output", default="output/eval_multi_res", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max-images", type=int, default=None, help="Limit images for quick test")
    parser.add_argument("--iou-types", nargs="+", default=["bbox", "segm"], help="COCO IoU types")
    parser.add_argument("--device", type=str, default=None, help="Device (default: from config)")
    parser.add_argument("--amp", action="store_true", default=False, help="Enable AMP")
    return parser.parse_args()


def build_val_loader(config, dataset_root_override=None, num_workers=4, batch_size=1):
    """Build validation dataloader with the image_size from config.data.image_size."""
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

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return dataset, loader


def evaluate_at_resolution(model, config, checkpoint_path, resolution,
                           dataset_root=None, output_dir=None,
                           batch_size=1, num_workers=4, max_images=None,
                           iou_types=None, amp_enabled=False, device=None):
    """Evaluate model at a specific resolution.

    The model itself is resolution-agnostic -- we only rebuild the dataloader
    with a different image_size transform. The same model weights are used
    across all resolutions.

    Returns:
        dict: COCO metrics (bbox_AP, segm_AP, etc.)
    """
    import copy
    cfg = copy.deepcopy(config)
    cfg.data.image_size = resolution

    if output_dir is None:
        output_dir = f"output/eval_multi_res/{resolution}px"

    dataset, loader = build_val_loader(
        cfg,
        dataset_root_override=dataset_root,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    if max_images is not None and max_images < len(loader):
        # run_inference_evaluation supports max_images
        pass

    if iou_types is None:
        iou_types = ["bbox", "segm"]

    result = run_inference_evaluation(
        model,
        loader,
        coco_gt=dataset.coco,
        device=device,
        output_dir=output_dir,
        amp_enabled=amp_enabled,
        iou_types=iou_types,
        max_images=max_images,
        fail_on_empty=True,
        category_ids=list(getattr(dataset, "category_ids", [])) or None,
    )

    return result.coco_metrics


def main():
    args = parse_args()

    resolutions = [int(r.strip()) for r in args.resolutions.split(",")]

    print(f"Multi-Resolution Evaluation")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config:     {args.config_file}")
    print(f"  Resolutions: {resolutions}")
    print(f"  IoU types:  {args.iou_types}")
    print()

    # Load config (resolution will be overridden per-run)
    overrides = {}
    if args.dataset_root is not None:
        overrides.setdefault("data", {})["dataset_root"] = args.dataset_root
    overrides.setdefault("model", {})["weights"] = args.checkpoint
    overrides.setdefault("runtime", {})["output_dir"] = args.output

    config = load_config(args.config_file, overrides=overrides)

    device = setup_device(config.runtime)
    if args.device is not None:
        device = torch.device(args.device)
    set_seed(config.runtime.seed)

    print(f"  Device: {device}")
    print()

    # Build model once -- it's resolution-agnostic
    print("Building model...")
    model = build_model(config)
    load_checkpoint(args.checkpoint, model, strict=True)
    model = model.to(device)
    model.eval()
    print(f"Model loaded and on {device}")
    print()

    # Evaluate at each resolution
    all_results = OrderedDict()
    output_base = Path(args.output)

    for res in resolutions:
        print(f"\n{'='*70}")
        print(f"  Evaluating at {res}px")
        print(f"{'='*70}")

        res_output = output_base / f"{res}px"

        t0 = time.time()
        try:
            metrics = evaluate_at_resolution(
                model=model,
                config=config,
                checkpoint_path=args.checkpoint,
                resolution=res,
                dataset_root=args.dataset_root,
                output_dir=str(res_output),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_images=args.max_images,
                iou_types=args.iou_types,
                amp_enabled=args.amp,
                device=device,
            )
            elapsed = time.time() - t0
            all_results[res] = metrics

            print(f"\n  [{res}px] Results ({elapsed:.1f}s):")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v:.4f}")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{res}px] ERROR ({elapsed:.1f}s): {e}")
            import traceback
            traceback.print_exc()
            all_results[res] = None

        # Free GPU memory between runs
        gc.collect()
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n\n{'='*70}")
    print("  SUMMARY: Results Across Resolutions")
    print(f"{'='*70}")

    # Collect all metric keys
    all_keys = []
    for res, metrics in all_results.items():
        if metrics is not None:
            for k in metrics:
                if k not in all_keys:
                    all_keys.append(k)

    # Print header
    header = f"{'Metric':<25s}"
    for res in resolutions:
        header += f" | {res}px"
    print(header)
    print("-" * len(header))

    for key in all_keys:
        row = f"{key:<25s}"
        for res in resolutions:
            metrics = all_results.get(res)
            if metrics is not None and key in metrics:
                val = metrics[key]
                if isinstance(val, (int, float)):
                    row += f" | {val:.4f}"
                else:
                    row += f" | {val}"
            else:
                row += f" | N/A"
        print(row)

    print(f"{'='*70}")

    # Save summary to file
    summary_path = output_base / "multi_res_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Config: {args.config_file}\n")
        f.write(f"Resolutions: {resolutions}\n\n")

        for res in resolutions:
            metrics = all_results.get(res)
            f.write(f"\n[{res}px]\n")
            if metrics is not None:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        f.write(f"  {k}: {v:.4f}\n")
            else:
                f.write("  FAILED\n")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
