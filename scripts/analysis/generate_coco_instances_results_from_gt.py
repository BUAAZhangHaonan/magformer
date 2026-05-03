#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pycocotools.coco import COCO


def _resolve_ann_path(dataset_root: Path, ann_file: str) -> Path:
    p = Path(ann_file)
    return p if p.is_absolute() else dataset_root / ann_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--ann-file", default="annotations/instances_val.json")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--limit-images", type=int, default=-1, help="-1 for all")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    ann_path = _resolve_ann_path(dataset_root, args.ann_file).resolve()
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    coco = COCO(str(ann_path))
    image_ids = sorted(coco.getImgIds())
    if int(args.limit_images) >= 0:
        image_ids = image_ids[: int(args.limit_images)]

    rows: List[Dict[str, Any]] = []
    for image_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=[int(image_id)])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            rle = coco.annToRLE(ann)
            counts = rle.get("counts")
            if isinstance(counts, bytes):
                rle["counts"] = counts.decode("ascii")
            rows.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(ann.get("category_id", 1)),
                    "score": 1.0,
                    "bbox": ann.get("bbox"),
                    "segmentation": rle,
                }
            )

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[gt->coco] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

