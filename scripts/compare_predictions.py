#!/usr/bin/env python3
"""
Generate GT / MagFormer / Mask2Former triptych visualizations from COCO results.
"""

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
sys.path.insert(0, str(Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from magformer.utils.visualization import (
    prediction_to_lists,
    render_triptych_comparison,
    visualize_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MagFormer and Mask2Former predictions")
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory")
    parser.add_argument("--ann-file", default="annotations/instances_val.json", help="COCO annotation file")
    parser.add_argument("--split", default="val", help="Image split directory (e.g. val/test)")
    parser.add_argument("--magformer-results", required=True, help="MagFormer COCO results json")
    parser.add_argument("--mask2former-results", required=True, help="Mask2Former COCO results json")
    parser.add_argument("--output-dir", default="output/comparison/visualizations", help="Output directory")
    parser.add_argument("--num-images", type=int, default=50, help="Number of images to visualize (-1 for all)")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Prediction score threshold")
    parser.add_argument("--alpha", type=float, default=0.3, help="Mask overlay alpha")
    parser.add_argument("--show-labels", action="store_true", help="Render score labels on overlays")
    return parser.parse_args()


def _resolve_ann_path(dataset_root: Path, ann_file: str) -> Path:
    ann_path = Path(ann_file)
    if ann_path.is_absolute():
        return ann_path
    return dataset_root / ann_file


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
    for path in candidates:
        if path.exists():
            return path
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
                # COCO result bbox convention: [x, y, w, h].
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

    return {
        "masks": masks,
        "scores": scores,
        "category_ids": labels,
    }


def _build_gt_masks(coco: COCO, image_id: int, height: int, width: int) -> List[np.ndarray]:
    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    masks: List[np.ndarray] = []
    for ann in anns:
        mask = _decode_segmentation(ann.get("segmentation", None), height, width)
        if np.any(mask):
            masks.append(mask)
    return masks


def _load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
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


def _save_single_overlay(
    image: np.ndarray,
    prediction: Dict[str, Any],
    output_path: Path,
    score_threshold: float,
    alpha: float,
    show_labels: bool,
) -> None:
    masks, scores, labels = prediction_to_lists(prediction)
    visualize_predictions(
        image=image,
        masks=masks,
        scores=scores,
        labels=labels,
        class_names=["component"],
        score_threshold=score_threshold,
        alpha=alpha,
        show_labels=show_labels,
        show_contours=True,
        contour_thickness=1,
        show_masks=True,
        output_path=str(output_path),
    )


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    ann_path = _resolve_ann_path(dataset_root, args.ann_file)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    output_dir = Path(args.output_dir)
    triptych_dir = output_dir / "triptych"
    mag_single_dir = output_dir / "magformer_single"
    m2f_single_dir = output_dir / "mask2former_single"
    triptych_dir.mkdir(parents=True, exist_ok=True)
    mag_single_dir.mkdir(parents=True, exist_ok=True)
    m2f_single_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(ann_path))
    image_ids = sorted(coco.getImgIds())
    if args.num_images >= 0:
        image_ids = image_ids[: args.num_images]

    mag_rows = _load_results(args.magformer_results)
    m2f_rows = _load_results(args.mask2former_results)
    mag_by_image = _group_by_image_id(mag_rows)
    m2f_by_image = _group_by_image_id(m2f_rows)

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

        gt_masks = _build_gt_masks(coco, image_id=image_id, height=h, width=w)
        mag_pred = _rows_to_prediction(mag_by_image.get(image_id, []), height=h, width=w)
        m2f_pred = _rows_to_prediction(m2f_by_image.get(image_id, []), height=h, width=w)

        triptych = render_triptych_comparison(
            image=image,
            gt_masks=gt_masks,
            magformer_prediction=mag_pred,
            mask2former_prediction=m2f_pred,
            score_threshold=float(args.score_threshold),
            alpha=float(args.alpha),
            show_labels=bool(args.show_labels),
            class_names=["component"],
            add_titles=True,
        )

        triptych_path = triptych_dir / f"triptych_{idx:04d}_id{image_id}.png"
        cv2.imwrite(str(triptych_path), cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR))

        _save_single_overlay(
            image=image,
            prediction=mag_pred,
            output_path=mag_single_dir / f"magformer_{idx:04d}_id{image_id}.png",
            score_threshold=float(args.score_threshold),
            alpha=float(args.alpha),
            show_labels=bool(args.show_labels),
        )
        _save_single_overlay(
            image=image,
            prediction=m2f_pred,
            output_path=m2f_single_dir / f"mask2former_{idx:04d}_id{image_id}.png",
            score_threshold=float(args.score_threshold),
            alpha=float(args.alpha),
            show_labels=bool(args.show_labels),
        )
        saved += 1

    print(f"[Compare] Saved {saved} triptych files to {triptych_dir}")
    print(f"[Compare] Saved MagFormer single overlays to {mag_single_dir}")
    print(f"[Compare] Saved Mask2Former single overlays to {m2f_single_dir}")


if __name__ == "__main__":
    main()
