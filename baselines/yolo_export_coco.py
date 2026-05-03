#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

from normalization_stats import load_dataset_normalization_stats
from yolo_stats_norm import StatsNormalizedSegmentationPredictor


def _find_weights(out_dir: Path) -> Optional[Path]:
    weights = out_dir / "train" / "weights"
    best = weights / "best.pt"
    last = weights / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    return None


def _load_coco_images(ann_file: Path) -> List[Dict[str, Any]]:
    d = json.loads(ann_file.read_text(encoding="utf-8"))
    images = d.get("images", [])
    return images


def _seg_to_rle(seg: Any, h: int, w: int) -> Optional[Dict[str, Any]]:
    # seg can be Nx2 array or list of Nx2 arrays. Convert to COCO RLE.
    polys: List[List[float]] = []
    if seg is None:
        return None
    if isinstance(seg, (list, tuple)):
        for s in seg:
            a = np.asarray(s, dtype=np.float32)
            if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] != 2:
                continue
            polys.append(a.reshape(-1).tolist())
    else:
        a = np.asarray(seg, dtype=np.float32)
        if a.ndim == 2 and a.shape[0] >= 3 and a.shape[1] == 2:
            polys.append(a.reshape(-1).tolist())

    if not polys:
        return None

    try:
        rles = mask_utils.frPyObjects(polys, h, w)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
    except Exception:
        return None

    # JSON-ify counts.
    if isinstance(rle.get("counts"), (bytes, bytearray)):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _import_ultralytics_yolo():
    """
    Import `ultralytics.YOLO` robustly.

    Note: This repo vendors the official Ultralytics git repo under
    `baselines/ultralytics/`. When executing this script from `baselines/`,
    `sys.path[0]` points to `baselines/` and may cause Python to treat
    `baselines/ultralytics/` as a *namespace* package and fail
    `from ultralytics import YOLO`.

    We fix this by prepending `baselines/ultralytics/` to `sys.path` so that
    Python resolves the actual package at `baselines/ultralytics/ultralytics/`.
    """
    import sys

    ultra_repo_root = Path(__file__).resolve().parent / "ultralytics"
    sys.path.insert(0, str(ultra_repo_root))

    from ultralytics import YOLO  # type: ignore

    return YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--ann-file", type=str, required=True)
    ap.add_argument("--split", type=str,
                    choices=["train", "val"], required=True)
    ap.add_argument("--weights", type=str, default="")
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--limit-images", type=int, default=0,
                    help="For smoke only; 0 means all.")
    ap.add_argument("--rgb-mean", type=str, default="")
    ap.add_argument("--rgb-std", type=str, default="")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    ann_file = Path(args.ann_file)
    out_json = Path(args.output_json)

    YOLO = _import_ultralytics_yolo()

    weights = Path(args.weights) if args.weights else _find_weights(
        out_json.parent)
    if weights is None or not weights.exists():
        raise SystemExit(
            f"[yolo-export] weights not found (pass --weights): {weights}")

    images = _load_coco_images(ann_file)
    if args.limit_images and args.limit_images > 0:
        images = images[: args.limit_images]

    img_paths: List[Path] = []
    for im in images:
        img_paths.append(dataset_root / "images" /
                         args.split / str(im["file_name"]))

    model = YOLO(str(weights))
    stats = load_dataset_normalization_stats(str(dataset_root))
    rgb_mean = json.loads(args.rgb_mean) if args.rgb_mean else list(stats.rgb_mean_rgb_255)
    rgb_std = json.loads(args.rgb_std) if args.rgb_std else list(stats.rgb_std_rgb_255)
    results_iter = model.predict(
        source=[str(p) for p in img_paths],
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        max_det=int(args.max_det),
        device=str(args.device),
        stream=True,
        verbose=False,
        predictor=StatsNormalizedSegmentationPredictor,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
    )

    coco_results: List[Dict[str, Any]] = []
    n_seen = 0
    for im, r in tqdm(zip(images, results_iter), total=len(img_paths), desc="[yolo-export] infer"):
        n_seen += 1
        image_id = int(im["id"])
        h = int(im["height"])
        w = int(im["width"])

        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)
        if boxes is None or masks is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        # cls is float tensor in ultralytics.
        cls = boxes.cls.cpu().numpy()

        segs = masks.xy  # pixel coords, already scaled to original image

        n = min(len(xyxy), len(segs))
        for i in range(n):
            rle = _seg_to_rle(segs[i], h, w)
            if rle is None:
                continue

            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
            bbox = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

            score = float(conf[i])
            _ = cls[i]  # reserved for future multi-class mapping

            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "segmentation": rle,
                    "score": score,
                }
            )

    if n_seen != len(img_paths):
        raise RuntimeError(
            f"[yolo-export] results length mismatch: got {n_seen}, expected {len(img_paths)}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco_results) + "\n", encoding="utf-8")
    print(f"[yolo-export] wrote: {out_json} (n={len(coco_results)})")


if __name__ == "__main__":
    main()
