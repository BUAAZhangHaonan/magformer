#!/usr/bin/env python3
"""
MAGFormer Training Diagnostic Script

诊断训练问题:
1. 检查数据加载是否正确
2. 检查ground truth masks是否有效
3. 检查模型前向传播
4. 检查Hungarian matcher输出
5. 检查loss计算
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np


def main():
    print("=" * 60)
    print("MAGFormer Training Diagnostic")
    print("=" * 60)

    # Check dataset
    dataset_root = "/home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0909_512_0.12K"

    if not os.path.exists(dataset_root):
        print(f"[ERROR] Dataset not found: {dataset_root}")
        print("Please specify the correct dataset path.")
        return

    print(f"\n[1] Checking dataset: {dataset_root}")

    # Load config
    from magformer.config import load_config
    config = load_config("configs/magformer_2k.yaml", overrides={"data": {"dataset_root": dataset_root}})

    # Build dataset
    from magformer.data import CocoRgbdDataset
    from magformer.data.transforms import RGBDTransform
    from magformer.data.collate import collate_fn
    from torch.utils.data import DataLoader

    train_dataset = CocoRgbdDataset(
        dataset_root=dataset_root,
        ann_file="annotations/instances_train.json",
        split="train",
        transform=None,
        is_train=True,
    )

    # Apply transform
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
        is_train=True,
    )
    train_dataset.transform = train_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"   Dataset size: {len(train_dataset)}")

    # Check a batch
    print("\n[2] Checking data batch...")
    batch = next(iter(train_loader))
    images = batch["images"]
    depths = batch["depths"]
    targets = batch.get("targets", [])

    print(f"   Images shape: {images.shape}, dtype: {images.dtype}, range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"   Depths shape: {depths.shape}, dtype: {depths.dtype}, range: [{depths.min():.4f}, {depths.max():.4f}]")

    # Check targets
    print("\n[3] Checking ground truth targets...")
    target_info = []
    for i, tgt in enumerate(targets):
        masks = tgt.get("masks", torch.zeros(0))
        labels = tgt.get("labels", torch.zeros(0))
        num_instances = len(labels)
        if num_instances > 0:
            mask_areas = [masks[j].sum().item() for j in range(num_instances)]
            avg_area = np.mean(mask_areas) if mask_areas else 0
        else:
            avg_area = 0
        target_info.append([f"Sample {i}", num_instances, f"{avg_area:.0f} px"])

    print("   " + "-" * 50)
    print(f"   {'Sample':<15} {'Num Instances':<15} {'Avg Mask Area':<15}")
    print("   " + "-" * 50)
    for row in target_info:
        print(f"   {row[0]:<15} {row[1]:<15} {row[2]:<15}")

    # Build model
    print("\n[4] Building model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    from magformer.models import build_model
    model = build_model(config)
    model = model.to(device)
    model.train()

    # Forward pass
    print("\n[5] Running forward pass...")
    images = images.to(device)
    depths = depths.to(device)

    # Move targets to device
    targets_device = []
    for tgt in targets:
        tgt_device = {k: v.to(device) if hasattr(v, "to") else v for k, v in tgt.items()}
        targets_device.append(tgt_device)

    # First get raw outputs by calling internal methods directly
    model.train()

    # Extract features and decoder outputs to check intermediate results
    with torch.no_grad():
        B, _, H, W = images.shape

        # Normalize inputs
        images_norm = (images - model.pixel_mean) / model.pixel_std

        # Extract multi-scale features
        rgb_features = model.rgb_backbone(images_norm)
        depth_features = model.depth_backbone(depths)

        print(f"   RGB features: {list(rgb_features.keys())}")
        print(f"   Depth features: {list(depth_features.keys())}")

        # Modality fusion
        fused_features, confidence_maps, fusion_losses = model.fusion(
            image_features=rgb_features,
            depth_features=depth_features,
            depth_raw=depths,
            rgb_image=images_norm,
        )

        print(f"   Fused features: {list(fused_features.keys())}")

        # Pixel decoder
        decoder_inputs = model.pixel_decoder(
            features=fused_features,
            confidence_maps=confidence_maps,
            depth_raw=depths,
        )

        # Transformer decoder
        outputs = model.decoder(
            memory=decoder_inputs["memory"],
            mask_features=decoder_inputs["mask_features"],
            multi_scale_features=decoder_inputs.get("multi_scale_features", None),
            multi_scale_pos=decoder_inputs.get("multi_scale_pos", None),
            pos_key=decoder_inputs.get("pos_key_list", None),
        )

    pred_logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]

    print(f"   pred_logits shape: {pred_logits.shape}, range: [{pred_logits.min():.4f}, {pred_logits.max():.4f}]")
    print(f"   pred_masks shape: {pred_masks.shape}, range: [{pred_masks.min():.4f}, {pred_masks.max():.4f}]")

    # Check mask logits distribution
    sigmoid_masks = torch.sigmoid(pred_masks)
    print(f"   sigmoid(pred_masks) range: [{sigmoid_masks.min():.4f}, {sigmoid_masks.max():.4f}]")
    print(f"   sigmoid(pred_masks) mean: {sigmoid_masks.mean():.4f}")

    # Check Hungarian matcher
    print("\n[6] Checking Hungarian matcher...")
    from magformer.models.common.matcher import HungarianMatcher
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_mask=1.0,
        cost_dice=1.0,
        num_points=12544,
    )

    indices = matcher(outputs, targets_device)
    print(f"   Number of batches: {len(indices)}")
    for i, (src_idx, tgt_idx) in enumerate(indices):
        print(f"   Batch {i}: matched {len(src_idx)} queries to {len(tgt_idx)} targets")

    # Check loss computation
    print("\n[7] Checking loss computation...")
    from magformer.models.common.criterion import SetCriterion

    weight_dict = {
        "loss_ce": config.model.magformer.mask_former.class_weight,
        "loss_mask": config.model.magformer.mask_former.mask_weight,
        "loss_dice": config.model.magformer.mask_former.dice_weight,
    }

    criterion = SetCriterion(
        num_classes=config.model.magformer.sem_seg_head.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=config.model.magformer.mask_former.no_object_weight,
        losses=["labels", "masks"],
        num_points=config.model.magformer.mask_former.train_num_points,
        oversample_ratio=config.model.magformer.mask_former.oversample_ratio,
        importance_sample_ratio=config.model.magformer.mask_former.importance_sample_ratio,
    )
    criterion = criterion.to(device)

    loss_dict = criterion(outputs, targets_device)

    print("   Loss values:")
    for k, v in loss_dict.items():
        print(f"      {k}: {v.item():.4f}")

    # Run a few training iterations
    print("\n[8] Running 10 training iterations...")

    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)

    loss_history = []
    mask_logit_history = []

    for step in range(10):
        optimizer.zero_grad()

        # Get new batch
        batch = next(iter(train_loader))
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        targets_device = []
        for tgt in batch.get("targets", []):
            tgt_device = {k: v.to(device) if hasattr(v, "to") else v for k, v in tgt.items()}
            targets_device.append(tgt_device)

        # Model returns loss dict in training mode
        loss_dict = model(images, depths, targets=targets_device)
        loss = loss_dict["total_loss"]

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        # Get mask logits by running forward again in eval mode (without gradients)
        model.eval()
        with torch.no_grad():
            B, _, H, W = images.shape
            images_norm = (images - model.pixel_mean) / model.pixel_std
            rgb_features = model.rgb_backbone(images_norm)
            depth_features = model.depth_backbone(depths)
            fused_features, _, _ = model.fusion(
                image_features=rgb_features,
                depth_features=depth_features,
                depth_raw=depths,
                rgb_image=images_norm,
            )
            decoder_inputs = model.pixel_decoder(
                features=fused_features,
                confidence_maps=None,
                depth_raw=depths,
            )
            outputs = model.decoder(
                memory=decoder_inputs["memory"],
                mask_features=decoder_inputs["mask_features"],
                multi_scale_features=decoder_inputs.get("multi_scale_features", None),
                multi_scale_pos=decoder_inputs.get("multi_scale_pos", None),
                pos_key=decoder_inputs.get("pos_key_list", None),
            )
            mask_logit_history.append(outputs["pred_masks"].mean().item())
        model.train()

        if step % 2 == 0:
            print(f"   Step {step}: loss={loss.item():.4f}, mask_logit_mean={mask_logit_history[-1]:.4f}")

    print(f"\n   Loss trend: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}")
    print(f"   Mask logit mean trend: {mask_logit_history[0]:.4f} -> {mask_logit_history[-1]:.4f}")

    # Check inference mode
    print("\n[9] Checking inference mode...")
    model.eval()

    with torch.no_grad():
        outputs = model(images, depths)

    if "predictions" in outputs:
        predictions = outputs["predictions"]
        print(f"   Number of batch predictions: {len(predictions)}")
        for i, pred in enumerate(predictions):
            scores = pred.get("scores", [])
            if len(scores) > 0:
                print(f"   Batch {i}: {len(scores)} predictions, top score: {scores[0]:.4f}")
            else:
                print(f"   Batch {i}: No predictions")
    else:
        print("   No predictions in output (expected in training mode without inference=True)")

    print("\n" + "=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)

    # Summary
    print("\n[Summary]")
    issues = []

    # Check if ground truth masks are present
    total_instances = sum(len(tgt.get("labels", [])) for tgt in targets)
    if total_instances == 0:
        issues.append("No ground truth instances found in the batch!")
    else:
        print(f"   Ground truth instances: OK ({total_instances} found)")

    # Check if matcher is matching
    total_matches = sum(len(src) for src, _ in indices)
    if total_matches == 0:
        issues.append("Hungarian matcher is not matching any queries to targets!")
    else:
        print(f"   Matcher: OK ({total_matches} matches)")

    # Check loss
    if loss_history[-1] < 1e-6:
        issues.append("Loss is too small, check criterion!")
    else:
        print(f"   Loss: OK (final={loss_history[-1]:.4f})")

    # Check mask logits
    if mask_logit_history[-1] < -5:
        issues.append("Mask logits are all very negative, model may not learn properly!")
    else:
        print(f"   Mask logits: OK (final mean={mask_logit_history[-1]:.4f})")

    if issues:
        print("\n[Issues Found]")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n   No major issues found. Training should work correctly.")


if __name__ == "__main__":
    main()
