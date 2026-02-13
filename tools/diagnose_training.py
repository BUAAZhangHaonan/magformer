#!/usr/bin/env python3
"""
MAGFormer Diagnostics (Mask2Former Parity)

目标:
- 验证 Val/Test 不做 resize/crop，预测 mask 与 COCO GT 尺寸严格一致（否则 COCOeval IoU=0, AP=0）。
- 验证 LSJ (ResizeScale + FixedSizeCrop) 语义与 detectron2 对齐，且 padding_mask 链路可用。
- 快速跑若干训练 iter，检查 loss/预测是否崩塌，并输出 GT vs Pred 可视化（YOLOv8 风格）。
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add magformer/ to sys.path for `import magformer.*`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _repo_root() -> Path:
    # .../electronic-components-grasp-and-segment/magformer/tools/diagnose_training.py -> repo root
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser("MAGFormer diagnostics")
    parser.add_argument(
        "--config",
        default=str(root / "magformer/configs/magformer_0831_debug.yaml"),
        help="Path to config yaml",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(root / "magformer_datasets/0831_1K"),
        help="Dataset root directory",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional checkpoint path (for inference/overlay checks)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "magformer/output/diagnostics"),
        help="Where to save debug visualizations",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=2,
        help="Batch size used for quick train-iter checks",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=10,
        help="Number of quick training iterations to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Score threshold for visualization",
    )
    parser.add_argument(
        "--val-index",
        type=int,
        default=0,
        help="Which val sample index to visualize/check",
    )
    return parser.parse_args()


def _move_targets_to_device(targets: List[Dict[str, Any]], device: torch.device) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in targets:
        t2 = {}
        for k, v in t.items():
            if torch.is_tensor(v):
                t2[k] = v.to(device)
            else:
                t2[k] = v
        out.append(t2)
    return out


def _to_uint8_rgb(image_chw: torch.Tensor) -> np.ndarray:
    img = image_chw.detach().cpu().float()
    if img.ndim != 3:
        raise ValueError(f"expected CHW image, got shape {tuple(img.shape)}")
    img = img.permute(1, 2, 0).numpy()
    if img.max() <= 1.0:
        img = (img * 255.0).clip(0, 255)
    img = img.clip(0, 255).astype(np.uint8)
    return img


def _gt_masks_to_list(gt_masks: torch.Tensor) -> List[np.ndarray]:
    if gt_masks.ndim != 3:
        return []
    out = []
    for i in range(gt_masks.shape[0]):
        m = gt_masks[i].detach().cpu().numpy()
        out.append((m > 0.5).astype(np.uint8) * 255)
    return out


def _pred_to_lists(pred: Dict[str, Any]) -> Tuple[List[np.ndarray], List[float], List[int]]:
    masks = pred.get("masks", None)
    scores = pred.get("scores", None)
    labels = pred.get("category_ids", None)

    if masks is None or scores is None:
        return [], [], []
    masks = np.asarray(masks)
    scores = np.asarray(scores).astype(np.float32).tolist()
    if labels is None:
        labels_list = [0] * len(scores)
    else:
        labels_list = np.asarray(labels).astype(np.int32).tolist()

    mask_list = []
    for i in range(masks.shape[0]):
        m = masks[i]
        if m.dtype != np.uint8:
            m = (m > 0.5).astype(np.uint8) * 255
        mask_list.append(m)
    return mask_list, scores, labels_list


def lsj_transform_sanity(
    sample: Dict[str, Any],
    image_size: int,
    min_scale: float,
    max_scale: float,
    num_trials: int = 50,
) -> None:
    from magformer.data.transforms import InitContentMask, ResizeScale, FixedSizeCrop

    rs = ResizeScale(min_scale=min_scale, max_scale=max_scale, target_size=image_size)
    crop = FixedSizeCrop((image_size, image_size), random_crop=False)

    scaled_hw: List[Tuple[int, int]] = []
    padded_ratio: List[float] = []

    keys = ("image", "depth", "masks", "boxes", "labels")
    base = {k: sample[k] for k in keys if k in sample}

    for _ in range(num_trials):
        # Cheap "deep copy" for arrays
        cur: Dict[str, Any] = {}
        for k, v in base.items():
            cur[k] = v.copy() if isinstance(v, np.ndarray) else v

        cur = InitContentMask()(cur)
        cur = rs(cur)
        h, w = cur["image"].shape[:2]
        scaled_hw.append((h, w))

        cur = crop(cur)
        cm = cur.get("content_mask", None)
        if cm is not None:
            cm = cm.astype(np.float32)
            padded_ratio.append(float(1.0 - cm.mean()))

    hs = [h for h, _ in scaled_hw]
    ws = [w for _, w in scaled_hw]
    print("[LSJ Sanity]")
    print(f"  target image_size: {image_size}, scale ~ U({min_scale}, {max_scale})")
    print(f"  resized H range: {min(hs)} .. {max(hs)} (mean={np.mean(hs):.1f})")
    print(f"  resized W range: {min(ws)} .. {max(ws)} (mean={np.mean(ws):.1f})")
    if padded_ratio:
        frac_padded = float(np.mean([r > 1e-6 for r in padded_ratio]))
        print(f"  padded after crop: {frac_padded*100:.1f}% samples (avg padded ratio={np.mean(padded_ratio):.3f})")


def main() -> None:
    args = parse_args()

    from magformer.config import load_config, setup_device, set_seed
    from magformer.data import CocoRgbdDataset
    from magformer.data.transforms import RGBDTransform
    from magformer.data.collate import collate_fn
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MAGFormer Diagnostics")
    print(f"  config: {args.config}")
    print(f"  dataset_root: {args.dataset_root}")
    print(f"  device: {device}")
    print(f"  torch: {torch.__version__}, cuda: {torch.version.cuda}, cudnn: {torch.backends.cudnn.version()}")
    print("=" * 80)

    # ---------------------------------------------------------------------
    # 1) LSJ transform sanity (train-side)
    # ---------------------------------------------------------------------
    raw_train = CocoRgbdDataset(
        dataset_root=args.dataset_root,
        ann_file=config.data.train_ann,
        split="train",
        transform=None,
        is_train=True,
    )
    raw_sample = raw_train[0]
    lsj_transform_sanity(
        raw_sample,
        image_size=int(config.data.image_size),
        min_scale=float(config.data.min_scale),
        max_scale=float(config.data.max_scale),
        num_trials=50,
    )

    # ---------------------------------------------------------------------
    # 2) Quick train loop (checks padding_masks + loss trend)
    # ---------------------------------------------------------------------
    train_dataset = CocoRgbdDataset(
        dataset_root=args.dataset_root,
        ann_file=config.data.train_ann,
        split="train",
        transform=None,
        is_train=True,
    )
    train_dataset.transform = RGBDTransform(
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
        depth_gaussian_std=config.data.depth_noise.gaussian_std,
        depth_speckle_std=config.data.depth_noise.speckle_std,
        depth_drop_prob=config.data.depth_noise.drop_prob,
        depth_drop_val=config.data.depth_noise.drop_val,
        is_train=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_model(config).to(device)
    if args.weights is not None:
        load_checkpoint(args.weights, model, strict=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.solver.base_lr),
        weight_decay=float(config.solver.weight_decay),
    )

    it = iter(train_loader)
    loss_hist: List[float] = []
    logit_mean_hist: List[float] = []

    if args.train_iters > 0:
        print("[Quick Train]")
        model.train()
        for step in range(args.train_iters):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            images = batch["images"].to(device, non_blocking=True)
            depths = batch["depths"].to(device, non_blocking=True)
            padding_masks = batch.get("padding_masks", None)
            if padding_masks is not None:
                padding_masks = padding_masks.to(device, non_blocking=True)

            targets = batch.get("targets", [])
            targets = _move_targets_to_device(targets, device)

            optimizer.zero_grad(set_to_none=True)
            losses = model(images, depths, targets=targets, padding_masks=padding_masks)
            total_loss = losses.get("total_loss", None)
            if total_loss is None:
                raise RuntimeError("model did not return total_loss in training mode")
            total_loss.backward()
            optimizer.step()

            loss_hist.append(float(total_loss.detach().cpu().item()))

            # monitor mask logit mean in eval mode (upsampled logits)
            with torch.no_grad():
                model.eval()
                out_eval = model(images, depths, padding_masks=padding_masks)
                pm = out_eval.get("pred_masks", None)
                if torch.is_tensor(pm):
                    logit_mean_hist.append(float(pm.float().mean().item()))
                else:
                    logit_mean_hist.append(float("nan"))
                model.train()

            if step == 0 or (step + 1) % 2 == 0:
                loss_mask = float(losses.get("loss_mask", torch.tensor(0.0)).detach().cpu().item()) if torch.is_tensor(losses.get("loss_mask", None)) else 0.0
                loss_dice = float(losses.get("loss_dice", torch.tensor(0.0)).detach().cpu().item()) if torch.is_tensor(losses.get("loss_dice", None)) else 0.0
                print(
                    f"  step {step+1:03d}/{args.train_iters} "
                    f"total={loss_hist[-1]:.3f} mask={loss_mask:.3f} dice={loss_dice:.3f} "
                    f"mask_logit_mean={logit_mean_hist[-1]:.3f}"
                )

        print(f"  loss trend: {loss_hist[0]:.3f} -> {loss_hist[-1]:.3f}")
        if logit_mean_hist:
            print(f"  mask_logit_mean trend: {logit_mean_hist[0]:.3f} -> {logit_mean_hist[-1]:.3f}")

    # ---------------------------------------------------------------------
    # 3) Val inference: mask size must match COCO GT size + GT vs Pred overlay
    # ---------------------------------------------------------------------
    val_dataset_gt = CocoRgbdDataset(
        dataset_root=args.dataset_root,
        ann_file=config.data.val_ann,
        split="val",
        transform=None,
        is_train=True,  # load GT masks for diagnostics
    )
    val_dataset_gt.transform = RGBDTransform(
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
        is_train=False,  # keep original resolution for eval sanity
    )

    idx = int(args.val_index) % len(val_dataset_gt)
    sample = val_dataset_gt[idx]
    img_id = int(sample.get("image_id", -1))
    img_info = val_dataset_gt.coco.loadImgs(img_id)[0]
    H_gt, W_gt = int(img_info["height"]), int(img_info["width"])
    H, W = int(sample["image"].shape[1]), int(sample["image"].shape[2])

    print("[Val Shape Check]")
    print(f"  img_id={img_id} coco_size=({H_gt},{W_gt}) tensor_size=({H},{W})")
    assert (H, W) == (H_gt, W_gt), f"val sample tensor size != COCO GT size: {(H,W)} vs {(H_gt,W_gt)}"

    model.eval()
    with torch.no_grad():
        images = sample["image"].unsqueeze(0).to(device, non_blocking=True)
        depths = sample["depth"].unsqueeze(0).to(device, non_blocking=True)
        out = model(images, depths)

    preds = out.get("predictions", [])
    if not preds:
        raise RuntimeError("model did not return predictions in eval mode")
    pred0 = preds[0]
    pred_masks = np.asarray(pred0.get("masks", []))
    if pred_masks.size > 0:
        assert pred_masks.shape[-2:] == (H_gt, W_gt), (
            f"pred mask size != COCO GT size: {tuple(pred_masks.shape[-2:])} vs {(H_gt,W_gt)}"
        )
    print(f"  pred_masks: {pred_masks.shape if pred_masks.size else '(empty)'}")

    # Save GT vs Pred overlay
    img_np = _to_uint8_rgb(sample["image"])
    masks_list, scores_list, labels_list = _pred_to_lists(pred0)
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

    gt_masks = sample.get("masks", None)
    if torch.is_tensor(gt_masks) and gt_masks.numel() > 0:
        gt_list = _gt_masks_to_list(gt_masks)
        for gm in gt_list:
            canvas = draw_yolov8_contour(canvas, gm, color=(0, 255, 0), thickness=2)

    # cv2.imwrite expects BGR
    import cv2

    out_path = out_dir / f"val_gt_pred_img{img_id}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"[Saved] {out_path}")

    print("=" * 80)
    print("Diagnostics complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
