#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Pseudo-Real Data for VC-SUDA Validation

Takes synthetic data and creates degraded "pseudo-real" versions
to validate the domain adaptation pipeline before real data is available.

Usage:
    python tools/prepare_pseudo_real.py \
        --dataset-root /path/to/synthetic \
        --output-root /path/to/pseudo_real \
        --source-ratio 0.8 \
        --labeled-ratio 0.02 \
        --seed 42
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare pseudo-real data from synthetic")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Root directory of synthetic dataset")
    parser.add_argument("--output-root", type=str, required=True,
                        help="Output directory for pseudo-real data")
    parser.add_argument("--source-ratio", type=float, default=0.8,
                        help="Fraction of images to keep as source (clean)")
    parser.add_argument("--labeled-ratio", type=float, default=0.02,
                        help="Fraction of images for target labeled set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy image files (default: symlink)")
    return parser.parse_args()


def add_poisson_gaussian_noise(image, sigma_range=(15, 25)):
    """Add realistic sensor noise."""
    sigma = random.uniform(*sigma_range)
    # Signal-dependent Poisson component
    poisson = np.random.poisson(image.astype(np.float32) * 0.1) / 0.1
    noisy = poisson + np.random.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def shift_color_temperature(image, kelvin_shift=2000):
    """Simulate color temperature shift."""
    result = image.astype(np.float32)
    shift = random.uniform(-kelvin_shift, kelvin_shift)
    # Warm shift: boost red, reduce blue. Cool: opposite.
    r_shift = shift / 6000.0 * 30
    b_shift = -shift / 6000.0 * 30
    result[:, :, 0] = np.clip(result[:, :, 0] + r_shift, 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] + b_shift, 0, 255)
    return result.astype(np.uint8)


def add_jpeg_artifacts(image, quality_range=(50, 70)):
    """Add JPEG compression artifacts."""
    quality = random.randint(*quality_range)
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def add_gaussian_blur(image, sigma_range=(1.0, 3.0)):
    """Add slight Gaussian blur."""
    sigma = random.uniform(*sigma_range)
    ksize = int(sigma * 4) * 2 + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def degrade_image(image, severity="medium"):
    """Apply full degradation pipeline to simulate real sensor gap."""
    result = image.copy()
    
    if severity == "light":
        result = add_poisson_gaussian_noise(result, sigma_range=(5, 10))
        result = shift_color_temperature(result, kelvin_shift=1000)
    elif severity == "medium":
        result = add_poisson_gaussian_noise(result, sigma_range=(15, 25))
        result = shift_color_temperature(result, kelvin_shift=2000)
        result = add_jpeg_artifacts(result, quality_range=(50, 70))
        result = add_gaussian_blur(result, sigma_range=(1.0, 2.0))
    elif severity == "heavy":
        result = add_poisson_gaussian_noise(result, sigma_range=(20, 40))
        result = shift_color_temperature(result, kelvin_shift=3000)
        result = add_jpeg_artifacts(result, quality_range=(30, 60))
        result = add_gaussian_blur(result, sigma_range=(1.5, 3.5))
    
    return result


def degrade_depth(depth, noise_std=0.02, hole_prob=0.05):
    """Add depth sensor noise and simulate missing depth holes."""
    result = depth.copy().astype(np.float32)
    # Gaussian noise
    result += np.random.normal(0, noise_std, result.shape)
    # Random holes (simulating depth sensor failures)
    hole_mask = np.random.random(result.shape) < hole_prob
    result[hole_mask] = 0.0
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def split_and_degrade(args):
    """Main function to split and degrade synthetic data."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    
    # Load original annotations
    ann_file = dataset_root / "annotations" / "instances_train.json"
    print(f"Loading annotations from: {ann_file}")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    all_images = coco_data["images"]
    all_annotations = coco_data["annotations"]
    
    # Build image_id -> [annotations] index for O(1) lookup
    ann_by_image = {}
    for ann in all_annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)
    
    # Shuffle and split
    random.shuffle(all_images)
    n_total = len(all_images)
    n_source = int(n_total * args.source_ratio)
    n_labeled = max(int(n_total * args.labeled_ratio), 1)
    n_unlabeled = n_total - n_source - n_labeled
    
    source_images = all_images[:n_source]
    labeled_images = all_images[n_source:n_source + n_labeled]
    unlabeled_images = all_images[n_source + n_labeled:]
    
    print(f"Split: source={len(source_images)}, labeled={len(labeled_images)}, "
          f"unlabeled={len(unlabeled_images)}")
    
    # Create output directories
    splits = {
        "source": source_images,
        "target_labeled": labeled_images,
        "target_unlabeled": unlabeled_images,
    }
    
    for split_name, split_images in splits.items():
        (output_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_root / "depth" / "depth_npy" / split_name).mkdir(parents=True, exist_ok=True)
    (output_root / "annotations").mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name, split_images in splits.items():
        is_target = split_name.startswith("target")
        severity = "medium" if is_target else None
        
        img_id_map = {}
        new_images = []
        new_annotations = []
        
        for img_info in split_images:
            old_id = img_info["id"]
            new_id = len(new_images) + 1
            img_id_map[old_id] = new_id
            
            filename = img_info["file_name"]
            
            # Copy/link image
            src_img = dataset_root / "images" / "train" / filename
            dst_img = output_root / "images" / split_name / filename
            
            if is_target and severity:
                # Degrade RGB image
                image = cv2.imread(str(src_img))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                degraded = degrade_image(image, severity=severity)
                cv2.imwrite(str(dst_img), cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR))
            else:
                if args.copy_images:
                    shutil.copy2(str(src_img), str(dst_img))
                else:
                    if dst_img.exists() or dst_img.is_symlink():
                        dst_img.unlink()
                    os.symlink(str(src_img.resolve()), str(dst_img))
            
            # Copy/link or degrade depth
            stem = Path(filename).stem
            src_depth = dataset_root / "depth" / "depth_npy" / "train" / f"{stem}.npy"
            dst_depth = output_root / "depth" / "depth_npy" / split_name / f"{stem}.npy"
            
            if is_target and src_depth.exists():
                depth = np.load(str(src_depth)).astype(np.float32)
                degraded_depth = degrade_depth(depth, noise_std=0.02, hole_prob=0.05)
                np.save(str(dst_depth), degraded_depth)
            elif src_depth.exists():
                if args.copy_images:
                    shutil.copy2(str(src_depth), str(dst_depth))
                else:
                    if dst_depth.exists() or dst_depth.is_symlink():
                        dst_depth.unlink()
                    os.symlink(str(src_depth.resolve()), str(dst_depth))
            
            # Build new image info
            new_img_info = {
                "id": new_id,
                "file_name": filename,
                "height": img_info.get("height", 512),
                "width": img_info.get("width", 512),
            }
            new_images.append(new_img_info)
            
            # Collect annotations for this image (index lookup)
            for ann in ann_by_image.get(old_id, []):
                new_ann = dict(ann)
                new_ann["id"] = len(new_annotations) + 1
                new_ann["image_id"] = new_id
                new_annotations.append(new_ann)
        
        # Write split annotation file
        split_coco = {
            "images": new_images,
            "annotations": new_annotations,
            "categories": coco_data["categories"],
        }
        ann_path = output_root / "annotations" / f"instances_{split_name}.json"
        with open(ann_path, 'w') as f:
            json.dump(split_coco, f)
        print(f"  {split_name}: {len(new_images)} images, {len(new_annotations)} annotations → {ann_path}")
    
    print(f"\nDone! Output at: {output_root}")
    print(f"Usage in config:")
    print(f"  data.dataset_root: {output_root}")
    print(f"  vc_suda.source_ann: annotations/instances_source.json")
    print(f"  vc_suda.target_labeled_ann: annotations/instances_target_labeled.json")
    print(f"  vc_suda.target_unlabeled_ann: annotations/instances_target_unlabeled.json")


if __name__ == "__main__":
    args = parse_args()
    split_and_degrade(args)
