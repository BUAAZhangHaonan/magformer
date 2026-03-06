#!/usr/bin/env python3
"""
Suite-level visualization (YOLOv8-seg style overlays, no class labels by default).

Outputs (fixed under output_root):
- visualizations/<model_id>/overlay/*.png
- visualizations/triptych_gt_magformer_mgm/*.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

try:
    from pycocotools.coco import COCO
except ModuleNotFoundError:
    class COCO:  # type: ignore[override]
        def __init__(self, annotation_file: str):
            payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
            self.dataset = payload
            self.imgs = {int(img["id"]): img for img in payload.get("images", [])}
            self.anns = {int(ann["id"]): ann for ann in payload.get("annotations", [])}
            self.img_to_anns = defaultdict(list)
            for ann in payload.get("annotations", []):
                self.img_to_anns[int(ann["image_id"])].append(int(ann["id"]))

        def getImgIds(self) -> List[int]:
            return sorted(self.imgs.keys())

        def getAnnIds(self, imgIds=None, iscrowd=None) -> List[int]:
            if imgIds is None:
                ids = list(self.anns.keys())
            elif isinstance(imgIds, list):
                ids = []
                for image_id in imgIds:
                    ids.extend(self.img_to_anns.get(int(image_id), []))
            else:
                ids = list(self.img_to_anns.get(int(imgIds), []))

            if iscrowd is None:
                return ids
            return [ann_id for ann_id in ids if int(self.anns[ann_id].get("iscrowd", 0)) == int(iscrowd)]

        def loadAnns(self, ids) -> List[Dict[str, Any]]:
            if isinstance(ids, list):
                return [self.anns[int(ann_id)] for ann_id in ids]
            return [self.anns[int(ids)]]


def _default_summary_path(output_root: Path) -> Path:
    return output_root / f"summary_{output_root.name}.json"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_model_entries(summary: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    ignore = {"experiment", "output_root"}
    out: List[Tuple[str, Dict[str, Any]]] = []
    for k, v in summary.items():
        if k in ignore:
            continue
        if not isinstance(v, dict):
            continue
        out.append((str(k), v))
    return out


def _resolve_ann_path(dataset_root: Path, ann_file: str) -> Path:
    p = Path(ann_file)
    return p if p.is_absolute() else dataset_root / ann_file


def _write_gt_results_json(coco: COCO, image_ids: List[int], out_path: Path) -> Path:
    rows: List[Dict[str, Any]] = []
    for image_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=[int(image_id)])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            seg = ann.get("segmentation")
            if seg is None:
                continue
            rows.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(ann.get("category_id", 1)),
                    "score": 1.0,
                    "bbox": ann.get("bbox"),
                    "segmentation": seg,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def _run_overlay(
    *,
    repo_root: Path,
    dataset_root: Path,
    ann_file: str,
    split: str,
    results_json: Path,
    output_dir: Path,
    num_images: int,
    score_threshold: float,
    alpha: float,
    show_labels: bool,
    prefix: str,
) -> None:
    vis_script = repo_root / "scripts" / "visualization" / "visualize_coco_results.py"
    if not vis_script.exists():
        raise FileNotFoundError(f"visualize script not found: {vis_script}")

    cmd = [
        sys.executable,
        str(vis_script),
        "--dataset-root",
        str(dataset_root),
        "--ann-file",
        str(ann_file),
        "--split",
        str(split),
        "--results-json",
        str(results_json),
        "--output-dir",
        str(output_dir),
        "--num-images",
        str(int(num_images)),
        "--score-threshold",
        str(float(score_threshold)),
        "--alpha",
        str(float(alpha)),
        "--prefix",
        str(prefix),
    ]
    if show_labels:
        cmd.append("--show-labels")
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _read_overlay(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _maybe_resize_like(img, ref):
    if img.shape[:2] == ref.shape[:2]:
        return img
    h, w = ref.shape[:2]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--summary", default="")
    ap.add_argument("--ann-file", default="annotations/instances_val.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-images", type=int, default=50)
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--show-labels", action="store_true")
    args = ap.parse_args()

    output_root = Path(args.output_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    summary_path = Path(args.summary).resolve() if args.summary else _default_summary_path(output_root)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary json not found: {summary_path}")

    repo_root = Path(__file__).resolve().parents[2]
    summary = _load_json(summary_path)

    ann_path = _resolve_ann_path(dataset_root, args.ann_file)
    coco = COCO(str(ann_path))
    image_ids = sorted(coco.getImgIds())
    if args.num_images >= 0:
        image_ids = image_ids[: int(args.num_images)]

    vis_root = output_root / "visualizations"
    vis_root.mkdir(parents=True, exist_ok=True)

    # 1) Per-model overlays
    for model_id, entry in _iter_model_entries(summary):
        if entry.get("status") not in {"ok", "partial"}:
            continue
        artifacts = entry.get("artifacts", {})
        if not isinstance(artifacts, dict):
            continue
        results = artifacts.get("coco_instances_results")
        if not results:
            continue
        results_path = Path(str(results))
        if not results_path.exists():
            continue

        out_dir = vis_root / model_id / "overlay"
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_overlay(
            repo_root=repo_root,
            dataset_root=dataset_root,
            ann_file=args.ann_file,
            split=args.split,
            results_json=results_path,
            output_dir=out_dir,
            num_images=len(image_ids),
            score_threshold=float(args.score_threshold),
            alpha=float(args.alpha),
            show_labels=bool(args.show_labels),
            prefix="overlay",
        )

    # 2) Triptych: GT / MAGFormer / MGM
    mag_dir = vis_root / "magformer" / "overlay"
    mgm_dir = vis_root / "mgm_mask2former" / "overlay"
    if mag_dir.exists() and mgm_dir.exists():
        gt_root = vis_root / "_gt"
        gt_results = _write_gt_results_json(coco, image_ids, gt_root / "gt_coco_instances_results.json")
        gt_overlay_dir = gt_root / "overlay"
        gt_overlay_dir.mkdir(parents=True, exist_ok=True)
        _run_overlay(
            repo_root=repo_root,
            dataset_root=dataset_root,
            ann_file=args.ann_file,
            split=args.split,
            results_json=gt_results,
            output_dir=gt_overlay_dir,
            num_images=len(image_ids),
            score_threshold=0.0,
            alpha=float(args.alpha),
            show_labels=False,
            prefix="overlay",
        )

        trip_dir = vis_root / "triptych_gt_magformer_mgm"
        trip_dir.mkdir(parents=True, exist_ok=True)
        for idx, image_id in enumerate(image_ids):
            gt_path = gt_overlay_dir / f"overlay_{idx:04d}_id{image_id}.png"
            mag_path = mag_dir / f"overlay_{idx:04d}_id{image_id}.png"
            mgm_path = mgm_dir / f"overlay_{idx:04d}_id{image_id}.png"
            if not (gt_path.exists() and mag_path.exists() and mgm_path.exists()):
                continue

            gt = _read_overlay(gt_path)
            mag = _maybe_resize_like(_read_overlay(mag_path), gt)
            mgm = _maybe_resize_like(_read_overlay(mgm_path), gt)
            stacked = cv2.hconcat([gt, mag, mgm])
            out_path = trip_dir / f"triptych_{idx:04d}_id{image_id}.png"
            cv2.imwrite(str(out_path), stacked)

    print(f"[visualize-suite] wrote overlays under: {vis_root}")


if __name__ == "__main__":
    main()
