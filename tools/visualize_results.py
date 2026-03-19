#!/usr/bin/env python3
"""
Visualize MAGFormer predictions with unified YOLOv8-seg style overlays.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from magformer.config import load_config
from magformer.models import build_model
from magformer.data import CocoRgbdDataset
from magformer.data.transforms import RGBDTransform
from magformer.data.collate import collate_fn
from magformer.utils.visualization import (
    prediction_to_lists,
    visualize_predictions as visualize_yolov8_predictions,
)


def _load_checkpoint(path: str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return {"state_dict": checkpoint}
    return checkpoint


def _checkpoint_to_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    if "model" in checkpoint:
        return checkpoint["model"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _extract_dataset_root_from_checkpoint(checkpoint: Dict[str, Any]) -> Optional[str]:
    cfg = checkpoint.get("config", None)
    if not isinstance(cfg, dict):
        return None
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        return None
    root = data_cfg.get("dataset_root", None)
    if isinstance(root, str) and root.strip():
        return root
    return None


def _extract_dataset_root_from_config_file(config_path: str) -> Optional[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        return None
    root = data_cfg.get("dataset_root", None)
    if isinstance(root, str) and root.strip():
        return root
    return None


def _resolve_eval_source(config: Any, split: str) -> Tuple[str, str]:
    """
    Resolve annotation file and data split for visualization.

    Args:
        config: Loaded config object
        split: "test" or "val"

    Returns:
        (ann_file, split_name)
    """
    if split == "val":
        return str(config.data.val_ann), str(config.data.val_split)

    # split == "test"
    test_ann = getattr(config.data, "test_ann", None)
    test_split = str(getattr(config.data, "test_split", "test"))
    dataset_root = Path(config.data.dataset_root)

    candidate_anns: List[str] = []
    if isinstance(test_ann, str) and test_ann.strip():
        candidate_anns.append(test_ann)
    candidate_anns.append("annotations/instances_test.json")

    for ann in candidate_anns:
        ann_path = Path(ann)
        if not ann_path.is_absolute():
            ann_path = (dataset_root / ann).resolve()
            if not ann_path.exists():
                ann_path = (dataset_root / "annotations" / Path(ann).name).resolve()
        if ann_path.exists():
            return ann, test_split

    raise FileNotFoundError(
        "Cannot resolve test annotation file. "
        "Tried config.data.test_ann and fallback annotations/instances_test.json. "
        "You can switch to --split val as a fallback."
    )


def _to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    """
    Convert CHW tensor to uint8 RGB without hard-coded mean/std assumptions.
    """
    img = image.detach().cpu().permute(1, 2, 0).numpy()
    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)
    if img.max() <= 1.5 and img.min() >= 0.0:
        img = img * 255.0
    img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


def _prepare_axes(num_samples: int, num_cols: int):
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(6 * num_cols, 4 * num_samples))
    if num_samples == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_samples == 1:
        axes = axes[np.newaxis, :]
    elif num_cols == 1:
        axes = axes[:, np.newaxis]
    return fig, axes


def visualize_predictions(
    model,
    dataset,
    output_dir: str,
    num_samples: int = 10,
    threshold: float = 0.5,
    alpha: float = 0.3,
    show_labels: bool = False,
    layout: str = "rgb_depth_overlay",
    save_all_samples: bool = True,
):
    """Visualize model predictions with unified YOLOv8-seg style rendering."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_count = min(int(num_samples), len(dataset))
    if grid_count < 0:
        raise ValueError("num_samples must be >= 0")

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    num_cols = 3 if layout == "rgb_depth_overlay" else 1
    fig = None
    axes = None
    if grid_count > 0:
        fig, axes = _prepare_axes(grid_count, num_cols)

    process_limit = len(dataset) if bool(save_all_samples) else grid_count

    with torch.no_grad():
        saved_count = 0
        for data_idx, batch in enumerate(loader):
            if data_idx >= process_limit:
                break

            images = batch["images"]
            depths = batch["depths"]
            image_id = int(batch["image_ids"][0].item())

            raw_output = model.forward_inference_raw(images, depths)
            output = model._export_inference_predictions(raw_output, include_raw_tensors=False)
            predictions = output.get("predictions", [])
            pred = predictions[0] if predictions else {}

            img = _to_uint8_rgb(images[0])
            depth = depths[0, 0].detach().cpu().numpy()
            masks, scores, labels = prediction_to_lists(pred)

            sample_path = None
            if save_all_samples or data_idx < grid_count:
                sample_path = output_dir / f"pred_{data_idx:03d}_id{image_id}_yolov8.png"
            overlay = visualize_yolov8_predictions(
                image=img,
                masks=masks,
                scores=scores,
                labels=labels,
                class_names=["component"],
                score_threshold=float(threshold),
                alpha=float(alpha),
                show_labels=bool(show_labels),
                show_contours=True,
                contour_thickness=1,
                show_masks=True,
                output_path=(str(sample_path) if sample_path is not None else None),
            )
            if sample_path is not None:
                saved_count += 1

            kept = sum(float(s) >= float(threshold) for s in scores)
            if data_idx < grid_count and axes is not None:
                if layout == "overlay":
                    axes[data_idx, 0].imshow(overlay)
                    axes[data_idx, 0].set_title(f"Overlay (ID: {image_id}, kept={kept})")
                    axes[data_idx, 0].axis("off")
                else:
                    axes[data_idx, 0].imshow(img)
                    axes[data_idx, 0].set_title(f"RGB (ID: {image_id})")
                    axes[data_idx, 0].axis("off")

                    axes[data_idx, 1].imshow(depth, cmap="viridis")
                    axes[data_idx, 1].set_title("Depth")
                    axes[data_idx, 1].axis("off")

                    axes[data_idx, 2].imshow(overlay)
                    axes[data_idx, 2].set_title(f"Overlay ({kept} objects)")
                    axes[data_idx, 2].axis("off")

    save_path = None
    if fig is not None:
        plt.tight_layout()
        save_path = output_dir / f"predictions_yolov8_{layout}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization grid to {save_path}")
    else:
        print("Skipped visualization grid because num_samples=0")

    print(f"Saved {saved_count} per-sample overlay images")
    return save_path


def plot_training_curves(log_file: str, output_dir: str):
    """Plot training loss curves from log file."""
    import re

    iterations = []
    losses = []

    with open(log_file, "r") as f:
        for line in f:
            match = re.search(r"iter=(\d+).*loss=([\d.]+)", line)
            if match:
                iterations.append(int(match.group(1)))
                losses.append(float(match.group(2)))

    if not iterations:
        print("No training data found in log file")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, losses, "b-", linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.grid(True, alpha=0.3)

    if len(losses) > 100:
        window = 100
        ma = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(iterations[window - 1 :], ma, "r-", linewidth=2, label=f"Moving Avg ({window})")
        ax.legend()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "training_loss_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curve to {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Visualize MAGFormer results")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--dataset-root", type=str, default=None, help="Dataset root")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="Output directory")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to include in the visualization grid",
    )
    parser.add_argument("--log-file", type=str, default=None, help="Training log file for loss curves")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for predictions")
    parser.add_argument("--alpha", type=float, default=0.3, help="Mask overlay alpha")
    parser.add_argument("--show-labels", action="store_true", help="Render class/score labels")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to visualize",
    )
    parser.add_argument(
        "--save-all-samples",
        dest="save_all_samples",
        action="store_true",
        help="Save per-sample overlays for all items in the selected split",
    )
    parser.add_argument(
        "--no-save-all-samples",
        dest="save_all_samples",
        action="store_false",
        help="Only save per-sample overlays for grid samples",
    )
    parser.set_defaults(save_all_samples=True)
    parser.add_argument(
        "--layout",
        type=str,
        default="rgb_depth_overlay",
        choices=["rgb_depth_overlay", "overlay"],
        help="Visualization layout",
    )
    args = parser.parse_args()

    checkpoint = None
    if args.checkpoint:
        checkpoint = _load_checkpoint(args.checkpoint)

    dataset_root = args.dataset_root or _extract_dataset_root_from_config_file(args.config)
    if not dataset_root and checkpoint is not None:
        dataset_root = _extract_dataset_root_from_checkpoint(checkpoint)
        if dataset_root:
            print(f"Using dataset root from checkpoint config: {dataset_root}")

    if not dataset_root:
        raise ValueError(
            "Dataset root is required. Provide --dataset-root, set data.dataset_root in config, "
            "or use a checkpoint with embedded config.data.dataset_root."
        )

    config = load_config(args.config, overrides={"data": {"dataset_root": dataset_root}})

    model = build_model(config)
    model.eval()

    if checkpoint is not None:
        state_dict = _checkpoint_to_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded checkpoint from {args.checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    ann_file, split_name = _resolve_eval_source(config, args.split)

    dataset = CocoRgbdDataset(
        dataset_root=config.data.dataset_root,
        ann_file=ann_file,
        split=split_name,
        transform=RGBDTransform(
            image_size=config.data.image_size,
            min_scale=config.data.min_scale,
            max_scale=config.data.max_scale,
            random_flip="none",
            depth_scale=config.data.depth.scale,
            depth_shift=config.data.depth.shift,
            depth_clip_min=config.data.depth.clip_min,
            depth_clip_max=config.data.depth.clip_max,
            depth_norm=config.data.depth.norm,
            depth_per_sample_norm=getattr(config.data.depth, "per_sample_norm", True),
            is_train=False,
        ),
        is_train=False,
    )

    visualize_predictions(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        threshold=args.threshold,
        alpha=args.alpha,
        show_labels=args.show_labels,
        layout=args.layout,
        save_all_samples=args.save_all_samples,
    )

    if args.log_file:
        plot_training_curves(args.log_file, args.output_dir)


if __name__ == "__main__":
    main()
