#!/usr/bin/env python3
"""
Overfit MAGFormer on a Single Image (Hard Sanity Check)

If the model cannot overfit 1 image (IoU/visualization stays bad), stop tuning hyperparams
and go back to checking geometry/mask semantics/losses.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add magformer/ to sys.path for `import magformer.*`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    p = argparse.ArgumentParser("overfit_one_image")
    p.add_argument(
        "--config",
        default=str(root / "magformer/configs/magformer_0831_debug.yaml"),
        help="Config yaml",
    )
    p.add_argument(
        "--dataset-root",
        default=str(root / "magformer_datasets/0831_1K"),
        help="Dataset root",
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "val"],
        help="Which split to overfit on",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index within the split",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Deterministic image_size for resize/crop (default: config.data.image_size)",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=800,
        help="Training iterations",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (default: config.solver.base_lr)",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Optional checkpoint to initialize from",
    )
    p.add_argument(
        "--output-dir",
        default=str(root / "magformer/output/overfit_one_image"),
        help="Where to save overlays",
    )
    p.add_argument(
        "--save-period",
        type=int,
        default=50,
        help="Save overlay every N iters",
    )
    p.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Score threshold for visualization/IoU",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return p.parse_args()


def _to_uint8_rgb(image_chw: torch.Tensor) -> np.ndarray:
    img = image_chw.detach().cpu().float().permute(1, 2, 0).numpy()
    if img.max() <= 1.0:
        img = (img * 255.0).clip(0, 255)
    return img.clip(0, 255).astype(np.uint8)


def _pred_to_lists(pred: Dict[str, Any]) -> Tuple[List[np.ndarray], List[float], List[int]]:
    masks = pred.get("masks", None)
    scores = pred.get("scores", None)
    labels = pred.get("category_ids", None)
    if masks is None or scores is None:
        return [], [], []
    masks = np.asarray(masks)
    scores_list = np.asarray(scores).astype(np.float32).tolist()
    if labels is None:
        labels_list = [0] * len(scores_list)
    else:
        labels_list = np.asarray(labels).astype(np.int32).tolist()
    out_masks: List[np.ndarray] = []
    for i in range(masks.shape[0]):
        m = masks[i]
        if m.dtype != np.uint8:
            m = (m > 0.5).astype(np.uint8) * 255
        out_masks.append(m)
    return out_masks, scores_list, labels_list


def _compute_best_iou(gt_masks: np.ndarray, pred_masks: np.ndarray) -> float:
    """
    gt_masks: (Ng, H, W) bool/0-1
    pred_masks: (Np, H, W) bool/0-1
    """
    if gt_masks.size == 0:
        return 0.0
    if pred_masks.size == 0:
        return 0.0

    gt = gt_masks.astype(bool)
    pred = pred_masks.astype(bool)

    bests: List[float] = []
    for g in gt:
        best = 0.0
        g_sum = int(g.sum())
        if g_sum == 0:
            bests.append(0.0)
            continue
        for p in pred:
            inter = int(np.logical_and(g, p).sum())
            union = int(g_sum + p.sum() - inter)
            if union <= 0:
                continue
            iou = float(inter) / float(union)
            if iou > best:
                best = iou
        bests.append(best)
    return float(np.mean(bests)) if bests else 0.0


def main() -> None:
    args = parse_args()

    from magformer.config import load_config, setup_device, set_seed
    from magformer.data import CocoRgbdDataset
    from magformer.data.collate import collate_fn
    from magformer.data.transforms import Compose, InitContentMask, ResizeScale, FixedSizeCrop, DepthNormalize, ToTensor
    from magformer.engine.utils import load_checkpoint
    from magformer.models import build_model
    from magformer.utils.visualization import visualize_predictions, draw_yolov8_contour

    overrides: Dict[str, Any] = {"data": {"dataset_root": args.dataset_root}}
    if args.weights is not None:
        overrides.setdefault("model", {})["weights"] = args.weights
    config = load_config(args.config, overrides=overrides)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = setup_device(config.runtime)

    image_size = int(args.image_size) if args.image_size is not None else int(config.data.image_size)
    lr = float(args.lr) if args.lr is not None else float(config.solver.base_lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic (no flip, no random crop): ResizeScale(scale=1) + center crop
    det_transform = Compose([
        InitContentMask(),
        ResizeScale(min_scale=1.0, max_scale=1.0, target_size=image_size),
        FixedSizeCrop((image_size, image_size), random_crop=False),
        DepthNormalize(
            scale=config.data.depth.scale,
            shift=config.data.depth.shift,
            clip_min=config.data.depth.clip_min,
            clip_max=config.data.depth.clip_max,
            norm=config.data.depth.norm,
        ),
        ToTensor(),
    ])

    ann_file = config.data.train_ann if args.split == "train" else config.data.val_ann
    dataset = CocoRgbdDataset(
        dataset_root=args.dataset_root,
        ann_file=ann_file,
        split=args.split,
        transform=det_transform,
        is_train=True,  # need GT for overfit
    )

    idx = int(args.index) % len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))

    images = batch["images"].to(device, non_blocking=True)
    depths = batch["depths"].to(device, non_blocking=True)
    padding_masks = batch.get("padding_masks", None)
    if padding_masks is not None:
        padding_masks = padding_masks.to(device, non_blocking=True)
    targets = batch.get("targets", [])
    targets_dev: List[Dict[str, Any]] = []
    for t in targets:
        t2 = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
        targets_dev.append(t2)

    # Keep GT masks on CPU for IoU
    gt_masks = targets[0].get("masks", torch.zeros(0)).detach().cpu().numpy().astype(bool)

    model = build_model(config).to(device)
    if args.weights is not None:
        load_checkpoint(args.weights, model, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(config.solver.weight_decay))

    print("=" * 80)
    print("Overfit One Image")
    print(f"  config: {args.config}")
    print(f"  dataset_root: {args.dataset_root}")
    print(f"  split/index: {args.split}/{idx}")
    print(f"  image_size: {image_size}")
    print(f"  device: {device}")
    print(f"  lr: {lr}")
    print(f"  iters: {args.iters}")
    print(f"  output_dir: {out_dir}")
    print("=" * 80)

    import cv2

    for it in range(1, int(args.iters) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = model(images, depths, targets=targets_dev, padding_masks=padding_masks)
        loss = losses["total_loss"]
        loss.backward()
        optimizer.step()

        if it == 1 or it % 10 == 0:
            loss_mask = float(losses.get("loss_mask", torch.tensor(0.0)).detach().cpu().item()) if torch.is_tensor(losses.get("loss_mask", None)) else 0.0
            loss_dice = float(losses.get("loss_dice", torch.tensor(0.0)).detach().cpu().item()) if torch.is_tensor(losses.get("loss_dice", None)) else 0.0
            print(f"  iter {it:04d}/{args.iters} total={float(loss.detach().cpu().item()):.3f} mask={loss_mask:.3f} dice={loss_dice:.3f}")

        if it == 1 or (args.save_period > 0 and it % args.save_period == 0) or it == int(args.iters):
            model.eval()
            with torch.no_grad():
                out = model(images, depths, padding_masks=padding_masks)
            preds = out.get("predictions", [])
            pred0 = preds[0] if preds else {}

            masks_list, scores_list, labels_list = _pred_to_lists(pred0)

            # IoU against GT (best IoU per GT instance, averaged)
            pred_probs = np.asarray(pred0.get("masks", []))
            pred_scores = np.asarray(pred0.get("scores", []), dtype=np.float32)
            if pred_probs.size == 0:
                best_iou = 0.0
            else:
                keep = pred_scores >= float(args.score_thr)
                pred_bin = (pred_probs[keep] > 0.5)
                best_iou = _compute_best_iou(gt_masks, pred_bin)

            img_np = _to_uint8_rgb(batch["images"][0])
            canvas = visualize_predictions(
                img_np,
                masks=masks_list,
                scores=scores_list,
                labels=labels_list,
                class_names=["component"],
                score_threshold=float(args.score_thr),
                alpha=0.4,
                show_labels=True,
                show_contours=True,
                show_masks=True,
                output_path=None,
            )
            # GT contours in green
            for g in gt_masks:
                gm = (g.astype(np.uint8) * 255)
                canvas = draw_yolov8_contour(canvas, gm, color=(0, 255, 0), thickness=2)

            out_path = out_dir / f"iter_{it:06d}_bestIoU_{best_iou:.3f}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            print(f"  [saved] {out_path}")

    print("=" * 80)
    print("Overfit done.")
    print("=" * 80)


if __name__ == "__main__":
    main()

