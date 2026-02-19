#!/usr/bin/env python3
"""
Single-model COCO results visualization (YOLOv8-seg style overlays).

Input:
- COCO annotation file
- COCO results json (list of dicts with image_id/category_id/score and segmentation RLE or polygons)

Output:
- Overlay PNGs under the given output directory
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

# Add project root to import path when invoked as a script.
sys.path.insert(0, str(Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from magformer.utils.visualization import prediction_to_lists, visualize_predictions


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--ann-file", default="annotations/instances_val.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-images", type=int, default=50, help="-1 for all")
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--show-labels", action="store_true")
    ap.add_argument("--prefix", type=str, default="overlay")
    return ap.parse_args()


def _resolve_ann_path(dataset_root: Path, ann_file: str) -> Path:
    p = Path(ann_file)
    return p if p.is_absolute() else dataset_root / ann_file


def _resolve_image_path(dataset_root: Path, split: str, file_name: str) -> Path:
    candidates = [
        dataset_root / "images" / split / file_name,
        dataset_root / "images" / file_name,
        dataset_root / split / file_name,
        dataset_root / file_name,
        dataset_root / "images" / split / Path(file_name).name,
        dataset_root / "images" / Path(file_name).name,
        dataset_root / split / Path(file_name).name,
        dataset_root / Path(file_name).name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _decode_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    if segmentation is None:
        return np.zeros((height, width), dtype=np.uint8)

    if isinstance(segmentation, list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        decoded = coco_mask.decode(rles)
    elif isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            rles = coco_mask.frPyObjects(segmentation, height, width)
            decoded = coco_mask.decode(rles)
        else:
            decoded = coco_mask.decode(segmentation)
    else:
        return np.zeros((height, width), dtype=np.uint8)

    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(np.uint8)


def _load_results(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"COCO results must be a list, got {type(data)}: {path}")
    return data


def _group_by_image_id(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        image_id = int(row.get("image_id", -1))
        if image_id < 0:
            continue
        grouped.setdefault(image_id, []).append(row)
    return grouped


def _rows_to_prediction(rows: List[Dict[str, Any]], height: int, width: int) -> Dict[str, Any]:
    masks: List[np.ndarray] = []
    scores: List[float] = []
    labels: List[int] = []

    for row in rows:
        score = float(row.get("score", 0.0))
        category_id = max(int(row.get("category_id", 1)) - 1, 0)

        seg = row.get("segmentation", None)
        mask = _decode_segmentation(seg, height, width)
        if not np.any(mask):
            bbox = row.get("bbox", None)
            if bbox is not None and len(bbox) == 4:
                # COCO bbox convention: [x, y, w, h]
                x, y, w, h = bbox
                x1 = max(int(round(x)), 0)
                y1 = max(int(round(y)), 0)
                x2 = min(int(round(x + w)), width)
                y2 = min(int(round(y + h)), height)
                if x2 > x1 and y2 > y1:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask[y1:y2, x1:x2] = 1

        if not np.any(mask):
            continue

        masks.append(mask)
        scores.append(score)
        labels.append(category_id)

    return {"masks": masks, "scores": scores, "category_ids": labels}


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    ann_path = _resolve_ann_path(dataset_root, args.ann_file)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    results_path = Path(args.results_json)
    if not results_path.exists():
        raise FileNotFoundError(f"Results json not found: {results_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(ann_path))
    image_ids = sorted(coco.getImgIds())
    if args.num_images >= 0:
        image_ids = image_ids[: args.num_images]

    rows = _load_results(results_path)
    by_image = _group_by_image_id(rows)

    saved = 0
    for idx, image_id in enumerate(image_ids):
        image_info = coco.loadImgs([image_id])[0]
        image_path = _resolve_image_path(dataset_root, args.split, image_info["file_name"])
        if not image_path.exists():
            print(f"[WARN] Skip missing image: {image_path}")
            continue

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            print(f"[WARN] Skip unreadable image: {image_path}")
            continue
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        pred = _rows_to_prediction(by_image.get(image_id, []), height=h, width=w)
        masks, scores, labels = prediction_to_lists(pred)
        vis = visualize_predictions(
            image=image,
            masks=masks,
            scores=scores,
            labels=labels,
            class_names=["component"],
            score_threshold=float(args.score_threshold),
            alpha=float(args.alpha),
            show_labels=bool(args.show_labels),
            show_contours=True,
            contour_thickness=1,
            show_masks=True,
        )

        out_path = out_dir / f"{args.prefix}_{idx:04d}_id{image_id}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        saved += 1

    print(f"[vis] saved {saved} overlays to {out_dir}")


if __name__ == "__main__":
    main()

