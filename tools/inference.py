#!/usr/bin/env python3
"""
MAGFormer Inference Script (Pure PyTorch)

Usage:
    python tools/inference.py --config-file configs/magformer.yaml --weights /path/to/ckpt.pth \
        --image /path/to/image.png --depth /path/to/depth.npy --output out.png
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from magformer.config import load_config, setup_device
from magformer.data.transforms import RGBDTransform
from magformer.models import build_model
from magformer.engine.utils import load_checkpoint
from magformer.utils.visualization import prediction_to_lists, visualize_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAGFormer inference")
    parser.add_argument("--config-file", required=True, help="Path to config yaml")
    parser.add_argument("--dataset-root", required=False, help="Override dataset root")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--image", required=True, help="RGB image path")
    parser.add_argument("--depth", required=True, help="Depth .npy/.npz path")
    parser.add_argument("--output", required=True, help="Output visualization path")
    return parser.parse_args()


def load_depth(path: str) -> np.ndarray:
    depth = np.load(path, allow_pickle=False)
    if isinstance(depth, np.lib.npyio.NpzFile):
        depth = depth[sorted(depth.files)[0]]
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    return depth.astype(np.float32)


def main() -> None:
    args = parse_args()

    overrides = {}
    if args.dataset_root is not None:
        overrides.setdefault("data", {})["dataset_root"] = args.dataset_root

    config = load_config(args.config_file, overrides=overrides)
    device = setup_device(config.runtime)

    model = build_model(config)
    load_checkpoint(args.weights, model, strict=False)
    model = model.to(device)
    model.eval()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    depth = load_depth(args.depth)

    transform = RGBDTransform(
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
        is_train=False,
    )

    sample = {"image": image, "depth": depth, "image_id": 0}
    sample = transform(sample)

    images = sample["image"].unsqueeze(0).to(device)
    depths = sample["depth"].unsqueeze(0).to(device)

    with torch.no_grad():
        raw_outputs = model.forward_inference_raw(images, depths)
        outputs = model._export_inference_predictions(
            raw_outputs,
            include_raw_tensors=False,
        )

    predictions = outputs.get("predictions", [])
    if not predictions:
        raise RuntimeError("No predictions returned by model")

    pred = predictions[0]
    masks, scores, labels = prediction_to_lists(pred)
    vis_image = visualize_predictions(
        image=image,
        masks=masks,
        scores=scores,
        labels=labels,
        class_names=["component"],
        score_threshold=0.5,
        alpha=0.3,
        show_labels=False,
        show_contours=True,
        contour_thickness=1,
        show_masks=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), vis_bgr)

    print(f"[Infer] Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
