#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    from pycocotools import mask as mask_utils
except Exception:  # pragma: no cover - optional during static checks
    mask_utils = None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def summarize_results(results_path: Path, image_area: float) -> Dict[str, Any]:
    rows = _load_json(results_path)
    bbox_ratios = []
    mask_areas = []
    scores = []
    for row in rows:
        bbox = row.get("bbox", None)
        if bbox is not None and len(bbox) == 4:
            _, _, bw, bh = bbox
            bbox_ratios.append((_safe_float(bw) * _safe_float(bh)) / max(1.0, image_area))
        score = row.get("score", None)
        if score is not None:
            scores.append(_safe_float(score))
        if mask_utils is not None and "segmentation" in row:
            area = mask_utils.area(row["segmentation"])
            if hasattr(area, "item"):
                area = area.item()
            mask_areas.append(float(area))

    bbox_arr = np.asarray(bbox_ratios, dtype=np.float64) if bbox_ratios else np.asarray([0.0], dtype=np.float64)
    mask_arr = np.asarray(mask_areas, dtype=np.float64) if mask_areas else np.asarray([0.0], dtype=np.float64)
    score_arr = np.asarray(scores, dtype=np.float64) if scores else np.asarray([0.0], dtype=np.float64)

    return {
        "pred_count": int(len(rows)),
        "bbox_area_ratio": {
            "p50": float(np.percentile(bbox_arr, 50)),
            "p90": float(np.percentile(bbox_arr, 90)),
            "max": float(np.max(bbox_arr)),
        },
        "mask_area": {
            "p50": float(np.percentile(mask_arr, 50)),
            "std": float(np.std(mask_arr)),
            "min": float(np.min(mask_arr)),
            "max": float(np.max(mask_arr)),
        },
        "score": {
            "p50": float(np.percentile(score_arr, 50)),
            "p90": float(np.percentile(score_arr, 90)),
            "max": float(np.max(score_arr)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json", type=str, required=True)
    ap.add_argument("--image-width", type=int, default=512)
    ap.add_argument("--image-height", type=int, default=512)
    ap.add_argument("--metrics-json", type=str, default="")
    ap.add_argument("--write", type=str, default="")
    args = ap.parse_args()

    results_json = Path(args.results_json).resolve()
    summary = summarize_results(
        results_path=results_json,
        image_area=float(int(args.image_width) * int(args.image_height)),
    )

    if args.metrics_json:
        metrics_path = Path(args.metrics_json).resolve()
        if metrics_path.exists():
            summary["metrics"] = _load_json(metrics_path)

    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.write:
        out = Path(args.write).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
        print(f"[audit] wrote: {out}")
    print(payload)


if __name__ == "__main__":
    main()
