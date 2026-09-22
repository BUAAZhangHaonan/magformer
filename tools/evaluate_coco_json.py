#!/usr/bin/env python3
"""Evaluate a standard COCO result JSON with the canonical metric contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pycocotools.coco import COCO

from magformer.engine.coco_json_eval import (
    SUPPORTED_IOU_TYPES,
    build_coco_eval_contract,
    evaluate_standard_coco_rows,
)



def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("COCO results JSON root must be an array")
    return payload


def evaluate_coco_json(
    *,
    annotation_path: Path,
    results_path: Path,
    expected_image_count: int | None = None,
    iou_types: tuple[str, ...] = SUPPORTED_IOU_TYPES,
) -> dict[str, Any]:
    coco_gt = COCO(str(annotation_path))
    contract = build_coco_eval_contract(coco_gt, max_dets=100)
    if expected_image_count is not None and len(contract.image_ids) != expected_image_count:
        raise ValueError(
            f"COCO GT image count mismatch: expected={expected_image_count}, "
            f"observed={len(contract.image_ids)}"
        )
    rows = _load_rows(results_path)
    metrics = evaluate_standard_coco_rows(
        coco_gt,
        rows,
        contract=contract,
        iou_types=iou_types,
    )
    return {
        "metric_scale": "fraction",
        "num_images": len(contract.image_ids),
        "num_predictions": len(rows),
        "contract": contract.to_dict(),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--expected-image-count", type=int)
    parser.add_argument(
        "--iou-types",
        nargs="+",
        choices=SUPPORTED_IOU_TYPES,
        default=list(SUPPORTED_IOU_TYPES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_path = args.ann_file.resolve()
    results_path = args.results_json.resolve()
    output_path = args.output_metrics.resolve()
    if output_path in {annotation_path, results_path}:
        raise ValueError("output metrics path must not overwrite an input JSON")
    payload = evaluate_coco_json(
        annotation_path=annotation_path,
        results_path=results_path,
        expected_image_count=args.expected_image_count,
        iou_types=tuple(args.iou_types),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[coco-json-eval] wrote: {output_path}")


if __name__ == "__main__":
    main()
