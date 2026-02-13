#!/usr/bin/env python3
"""
MAGFormer Evaluation Script (Pure PyTorch)

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
from magformer.engine.evaluator import COCOEvaluator
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

    model = build_model(config)
    if args.weights:
        load_checkpoint(args.weights, model, strict=False)
    model = model.to(device)
    model.eval()

    evaluator = COCOEvaluator(dataset.coco, iou_types=["bbox", "segm"])

    results = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            depths = batch["depths"].to(device)
            image_ids = batch["image_ids"].tolist()

            outputs = model(images, depths)
            predictions = outputs.get("predictions", [])

            for b_idx, pred in enumerate(predictions):
                img_id = image_ids[b_idx]
                scores = pred.get("scores", [])
                masks = pred.get("masks", [])
                category_ids = pred.get("category_ids", None)

                for s_idx, score in enumerate(scores):
                    mask = masks[s_idx]
                    if isinstance(mask, torch.Tensor):
                        mask = mask.detach().cpu().numpy()
                    mask = (mask > 0.5).astype("uint8")

                    if category_ids is not None:
                        category_id = int(category_ids[s_idx]) + 1
                    else:
                        category_id = 1

                    results.append({
                        "image_id": img_id,
                        "category_id": category_id,
                        "score": float(score),
                        "mask": mask,
                    })

    evaluator.update(results)
    metrics = evaluator.summarize()

    coco_results = evaluator._convert_to_coco_format(results)
    output_file = output_dir / "coco_instances_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        import json
        json.dump(coco_results, f)

    print(f"[Eval] Results saved to {output_file}")
    print(metrics)


if __name__ == "__main__":
    main()
