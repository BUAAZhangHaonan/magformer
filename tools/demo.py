#!/usr/bin/env python3
"""
MAGFormer Demo Script

This script runs MAGFormer inference on images or videos for visualization.

Usage:
    python demo.py --config configs/mgm_swin_convnext.yaml --model-path output/model_final.pth --input images/
"""

import os
import sys
import glob
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="MAGFormer Demo")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image/video/directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--depth-dir", type=str, default=None,
                        help="Directory containing depth maps")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for detections")
    return parser.parse_args()


def setup_model(args):
    """Setup MAGFormer model for inference."""
    import torch
    from detectron2.config import get_cfg
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model

    from mask2former import add_mgm_config

    # Load config
    cfg = get_cfg()
    add_mgm_config(cfg)
    cfg.merge_from_file(args.config)

    # Set model weights
    cfg.MODEL.WEIGHTS = args.model_path

    # Set confidence threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence

    # Build model
    model = build_model(cfg)
    model.eval()
    model.to(cfg.MODEL.DEVICE)

    # Load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(args.model_path)

    return cfg, model


def process_image(img_path, depth_path, cfg, model):
    """Process a single image.

    Args:
        img_path: Path to RGB image
        depth_path: Path to depth map (optional)
        cfg: Configuration
        model: MAGFormer model

    Returns:
        Processed image with visualization
    """
    from detectron2.data.detection_utils import read_image
    from detectron2.modeling.postprocessing import sem_seg_postprocess
    import torch

    # Read image
    image = read_image(img_path, format="BGR")
    original_height, original_width = image.shape[:2]

    # Read depth if available
    depth = None
    if depth_path and os.path.exists(depth_path):
        depth = np.load(depth_path)
        if depth.ndim == 3:
            depth = depth.squeeze()

    # Prepare input
    from detectron2.data import transforms as T
    from detectron2.data.datasets.augmentation import AugmentationList

    augmentations = AugmentationList([
        T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST,
        ),
    ])

    with torch.no_grad():
        # Apply augmentation
        original_image = image.copy()
        height, width = original_image.shape[:2]
        aug_input = T.AugInput(image, depth=depth)
        transform = augmentations(aug_input)
        image = aug_input.image
        if depth is not None:
            depth = aug_input.depth

        # Create input dict
        inputs = {
            "image": torch.as_tensor(image.astype("float32").transpose(2, 0, 1)),
            "height": height,
            "width": width,
        }

        if depth is not None:
            inputs["depth"] = torch.as_tensor(depth.astype("float32"))

        # Run model
        outputs = model([inputs])[0]

    # Visualize
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog

    # Get metadata
    if cfg.DATASETS.TEST:
        meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    else:
        meta = MetadataCatalog.get("eccd_val")

    v = Visualizer(
        original_image[:, :, ::-1],  # BGR to RGB
        metadata=meta,
        scale=1.0,
        instance_mode=ColorMode.IMAGE_BW
    )

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result = v.get_image()[:, :, ::-1]  # RGB to BGR

    return result


def main():
    """Main demo function."""
    import glob

    args = parse_args()

    # Setup model
    cfg, model = setup_model(args)

    # Setup output directory
    if args.output is None:
        output_dir = Path(args.input).parent / "output"
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get input files
    input_path = Path(args.input)

    if input_path.is_file():
        input_files = [input_path]
    else:
        input_files = sorted(input_path.glob("*.*jpg")) + sorted(input_path.glob("*.*png"))

    # Process each file
    for img_path in input_files:
        print(f"Processing: {img_path.name}")

        # Find corresponding depth
        depth_path = None
        if args.depth_dir:
            depth_dir = Path(args.depth_dir)
            depth_path = depth_dir / f"{img_path.stem}.npy"
            if not depth_path.exists():
                depth_path = depth_dir / f"{img_path.stem}.png"

        # Process
        result = process_image(str(img_path), str(depth_path) if depth_path else None, cfg, model)

        # Save result
        output_file = output_dir / f"output_{img_path.name}"
        cv2.imwrite(str(output_file), result)
        print(f"Saved: {output_file}")

    print(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
