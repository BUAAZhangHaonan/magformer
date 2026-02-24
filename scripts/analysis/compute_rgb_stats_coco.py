#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def _iter_image_paths(img_dir: Path, max_images: int | None) -> List[Path]:
    paths = sorted([p for p in img_dir.iterdir() if p.is_file()])
    if max_images is not None:
        paths = paths[: int(max_images)]
    return paths


def compute_rgb_stats(img_dir: Path, max_images: int | None = None) -> Dict[str, float | int | str | list[float]]:
    paths = _iter_image_paths(img_dir, max_images=max_images)
    if not paths:
        raise FileNotFoundError(f"No images found under: {img_dir}")

    sum_c = np.zeros((3,), dtype=np.float64)
    sumsq_c = np.zeros((3,), dtype=np.float64)
    num_pixels = 0

    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            a = np.asarray(im, dtype=np.float32)  # (H,W,3) RGB 0..255
        pixels = a.reshape(-1, 3).astype(np.float64)
        sum_c += pixels.sum(axis=0)
        sumsq_c += (pixels * pixels).sum(axis=0)
        num_pixels += int(pixels.shape[0])

    mean_rgb = sum_c / float(num_pixels)
    var_rgb = sumsq_c / float(num_pixels) - mean_rgb * mean_rgb
    var_rgb = np.maximum(var_rgb, 0.0)
    std_rgb = np.sqrt(var_rgb)

    mean_bgr = mean_rgb[::-1]
    std_bgr = std_rgb[::-1]

    return {
        "mean_rgb": [float(x) for x in mean_rgb.tolist()],
        "std_rgb": [float(x) for x in std_rgb.tolist()],
        "mean_bgr": [float(x) for x in mean_bgr.tolist()],
        "std_bgr": [float(x) for x in std_bgr.tolist()],
        "num_images": int(len(paths)),
        "num_pixels": int(num_pixels),
        "script_version": "v1",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True, help="Path to magformer_datasets/<dataset> root.")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--max-images", type=int, default=None, help="Optional cap for faster approximate stats.")
    ap.add_argument("--output", type=str, default=None, help="Optional output JSON path.")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    img_dir = dataset_root / "images" / str(args.split)
    stats = compute_rgb_stats(img_dir=img_dir, max_images=args.max_images)

    if args.output:
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[rgb-stats] wrote: {out}")

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

