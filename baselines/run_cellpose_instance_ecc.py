#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from cellpose_instance_models import run_experiment


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--max-train-steps", type=int, default=0)
    ap.add_argument("--max-val-images", type=int, default=0)
    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--val-split", type=str, default="val")
    args = ap.parse_args()
    if int(args.image_size) not in {512, 1024}:
        raise ValueError("--image-size must be one of {512, 1024}")
    return args


def main() -> None:
    args = _parse_args()
    run_experiment(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        num_workers=args.num_workers,
        min_area=args.min_area,
        device=args.device,
        max_train_steps=args.max_train_steps,
        max_val_images=args.max_val_images,
        train_split=args.train_split,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
