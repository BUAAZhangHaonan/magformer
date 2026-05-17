#!/usr/bin/env python3
"""Compute R78 target bucket AP and oracle recall from existing COCO predictions."""

from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRIC_KEYS = ("AP", "AP50", "AP75")


class DiagnosisError(RuntimeError):
    """Raised when a required diagnostic input is missing or malformed."""


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise DiagnosisError(f"missing input file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_coco(coco: Any, path: Path) -> dict[str, Any]:
    if not isinstance(coco, dict):
        raise DiagnosisError(f"{path} must be a COCO annotation object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(coco.get(key), list):
            raise DiagnosisError(f"{path} is missing COCO list field: {key}")
    return coco


def _require_predictions(predictions: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(predictions, list):
        raise DiagnosisError(f"{path} must be a COCO results list")
    for idx, item in enumerate(predictions):
        if not isinstance(item, dict):
            raise DiagnosisError(f"{path}[{idx}] must be an object")
        for key in ("image_id", "category_id", "bbox", "segmentation", "score"):
            if key not in item:
                raise DiagnosisError(f"{path}[{idx}] is missing required field: {key}")
    return predictions


def _mask_rle(segmentation: Any, height: int, width: int) -> dict[str, Any]:
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        return rle
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    raise DiagnosisError(f"unsupported segmentation type: {type(segmentation).__name__}")


def _annotation_area(ann: dict[str, Any], images: dict[int, dict[str, Any]]) -> float:
    area = ann.get("area")
    if area is not None:
        return float(area)
    image = images[int(ann["image_id"])]
    return float(mask_utils.area(_mask_rle(ann["segmentation"], int(image["height"]), int(image["width"]))))


def _mean_coco_metrics(eval_obj: COCOeval) -> dict[str, float]:
    precision = eval_obj.eval["precision"]
    area_idx = list(eval_obj.params.areaRngLbl).index("all")
    max_det_idx = list(eval_obj.params.maxDets).index(200)

    def mean_precision(iou_thr: float | None = None) -> float:
        if iou_thr is None:
            values = precision[:, :, :, area_idx, max_det_idx]
        else:
            idx = np.where(np.isclose(eval_obj.params.iouThrs, iou_thr))[0]
            values = precision[idx, :, :, area_idx, max_det_idx]
        values = values[values > -1]
        return float(np.mean(values)) if values.size else -1.0

    return {
        "AP": mean_precision(),
        "AP50": mean_precision(0.50),
        "AP75": mean_precision(0.75),
    }


def _eval_subset(
    coco: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    image_ids: set[int],
    positive_ann_ids: set[int],
    iou_type: str,
) -> dict[str, float]:
    if not image_ids:
        raise DiagnosisError("cannot evaluate an empty image subset")
    if not positive_ann_ids:
        raise DiagnosisError("cannot evaluate a subset with zero positive annotations")

    subset_images = [copy.deepcopy(image) for image in coco["images"] if int(image["id"]) in image_ids]
    subset_anns = []
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in image_ids:
            continue
        item = copy.deepcopy(ann)
        if int(item["id"]) in positive_ann_ids:
            item["ignore"] = 0
            item["iscrowd"] = int(item.get("iscrowd", 0))
        else:
            item["ignore"] = 1
            item["iscrowd"] = 1
        subset_anns.append(item)

    if not any(int(ann["id"]) in positive_ann_ids for ann in subset_anns):
        raise DiagnosisError("positive annotations are not present in the requested images")

    subset_predictions = [pred for pred in predictions if int(pred["image_id"]) in image_ids]
    if not subset_predictions:
        raise DiagnosisError("subset has no predictions")

    payload = {
        "images": subset_images,
        "annotations": subset_anns,
        "categories": copy.deepcopy(coco["categories"]),
        "info": copy.deepcopy(coco.get("info", {})),
        "licenses": copy.deepcopy(coco.get("licenses", [])),
    }
    with tempfile.TemporaryDirectory(prefix="r78_bucket_coco_") as tmpdir:
        ann_path = Path(tmpdir) / "ann.json"
        pred_path = Path(tmpdir) / "pred.json"
        ann_path.write_text(json.dumps(payload), encoding="utf-8")
        pred_path.write_text(json.dumps(subset_predictions), encoding="utf-8")
        coco_gt = COCO(str(ann_path))
        coco_dt = coco_gt.loadRes(str(pred_path))
        eval_obj = COCOeval(coco_gt, coco_dt, iou_type)
        eval_obj.params.maxDets = [1, 10, 200]
        eval_obj.evaluate()
        eval_obj.accumulate()
    return _mean_coco_metrics(eval_obj)


def _oracle_recall(
    coco: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    image_ids: set[int],
    positive_ann_ids: set[int],
) -> dict[str, float]:
    images = {int(image["id"]): image for image in coco["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    preds_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        if int(ann["image_id"]) in image_ids and int(ann["id"]) in positive_ann_ids:
            anns_by_image[int(ann["image_id"])].append(ann)
    for pred in predictions:
        if int(pred["image_id"]) in image_ids:
            preds_by_image[int(pred["image_id"])].append(pred)

    max_ious: list[float] = []
    for image_id in sorted(image_ids):
        anns = anns_by_image.get(image_id, [])
        if not anns:
            continue
        image = images[image_id]
        height = int(image["height"])
        width = int(image["width"])
        gt_rles = [_mask_rle(ann["segmentation"], height, width) for ann in anns]
        pred_rles = [_mask_rle(pred["segmentation"], height, width) for pred in preds_by_image.get(image_id, [])]
        if not pred_rles:
            max_ious.extend([0.0] * len(gt_rles))
            continue
        ious = mask_utils.iou(pred_rles, gt_rles, [int(ann.get("iscrowd", 0)) for ann in anns]).T
        max_ious.extend(np.max(ious, axis=1).astype(float).tolist())

    if len(max_ious) != len(positive_ann_ids):
        raise DiagnosisError(
            f"oracle accounting mismatch: got {len(max_ious)} IoUs for {len(positive_ann_ids)} annotations"
        )
    arr = np.asarray(max_ious, dtype=np.float64)
    return {
        "R50": float(np.mean(arr >= 0.50)),
        "R75": float(np.mean(arr >= 0.75)),
    }


def _load_r52_buckets(path: Path, expected_image_ids: set[int]) -> dict[str, set[int]]:
    payload = _load_json(path)
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        raise DiagnosisError(f"{path} must contain an images list")
    buckets = {"normal": set(), "dense": set(), "dense_tiny": set()}
    seen = set()
    for idx, row in enumerate(images):
        image_id = int(row["image_id"])
        bucket = row.get("bucket")
        if bucket not in buckets:
            raise DiagnosisError(f"{path}.images[{idx}] has unsupported bucket: {bucket}")
        if image_id in seen:
            raise DiagnosisError(f"{path} repeats image_id {image_id}")
        seen.add(image_id)
        buckets[bucket].add(image_id)
    if seen != expected_image_ids:
        missing = sorted(expected_image_ids - seen)[:10]
        extra = sorted(seen - expected_image_ids)[:10]
        raise DiagnosisError(f"R52 stats image ids differ from GT; missing={missing}, extra={extra}")
    return buckets


def _collect_bucket_defs(
    coco: dict[str, Any],
    r52_buckets: dict[str, set[int]],
    *,
    bottom_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images = {int(image["id"]): image for image in coco["images"]}
    anns = [ann for ann in coco["annotations"] if int(ann.get("iscrowd", 0)) == 0]
    ann_area = {int(ann["id"]): _annotation_area(ann, images) for ann in anns}
    all_image_ids = set(images)
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in anns:
        anns_by_image[int(ann["image_id"])].append(ann)

    def image_bucket(name: str, image_ids: set[int]) -> dict[str, Any]:
        ann_ids = {int(ann["id"]) for image_id in image_ids for ann in anns_by_image.get(image_id, [])}
        return {"name": name, "image_ids": image_ids, "positive_ann_ids": ann_ids}

    tiny_ann_ids = {ann_id for ann_id, area in ann_area.items() if area <= 256.0}
    sorted_anns = sorted(((area, int(ann["id"])) for ann in anns for area in [ann_area[int(ann["id"])]]))
    bottom_count = int(math.floor(len(sorted_anns) * bottom_fraction))
    if bottom_count <= 0:
        raise DiagnosisError("bottom target bucket is empty")
    bottom_ann_ids = {ann_id for _area, ann_id in sorted_anns[:bottom_count]}
    bottom_threshold = float(sorted_anns[bottom_count - 1][0])

    buckets = [
        image_bucket("overall", all_image_ids),
        image_bucket("normal", r52_buckets["normal"]),
        image_bucket("dense", r52_buckets["dense"]),
        image_bucket("dense_tiny", r52_buckets["dense_tiny"]),
        {"name": "tiny_area_le_256", "image_ids": all_image_ids, "positive_ann_ids": tiny_ann_ids},
        {"name": "bottom20_area", "image_ids": all_image_ids, "positive_ann_ids": bottom_ann_ids},
    ]
    metadata = {
        "bottom_fraction": bottom_fraction,
        "bottom20_area_threshold": bottom_threshold,
        "area_count": len(sorted_anns),
    }
    return buckets, metadata


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    ann_path = Path(args.ann)
    pred_path = Path(args.pred)
    stats_path = Path(args.r52_stats)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md) if args.out_md else None

    coco = _require_coco(_load_json(ann_path), ann_path)
    predictions = _require_predictions(_load_json(pred_path), pred_path)
    image_ids = {int(image["id"]) for image in coco["images"]}
    r52_buckets = _load_r52_buckets(stats_path, image_ids)
    bucket_defs, metadata = _collect_bucket_defs(coco, r52_buckets, bottom_fraction=float(args.bottom_fraction))

    rows = []
    for bucket in bucket_defs:
        name = bucket["name"]
        image_subset = set(bucket["image_ids"])
        ann_subset = set(bucket["positive_ann_ids"])
        bbox = _eval_subset(coco, predictions, image_ids=image_subset, positive_ann_ids=ann_subset, iou_type="bbox")
        segm = _eval_subset(coco, predictions, image_ids=image_subset, positive_ann_ids=ann_subset, iou_type="segm")
        oracle = _oracle_recall(coco, predictions, image_ids=image_subset, positive_ann_ids=ann_subset)
        rows.append(
            {
                "bucket": name,
                "images": len({int(ann["image_id"]) for ann in coco["annotations"] if int(ann["id"]) in ann_subset}),
                "annotations": len(ann_subset),
                "bbox": bbox,
                "segm": segm,
                "mask_oracle": oracle,
            }
        )

    payload = {
        "annotation_path": str(ann_path),
        "prediction_path": str(pred_path),
        "r52_stats_path": str(stats_path),
        "metadata": metadata,
        "buckets": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.expected_metrics:
        expected = _load_json(Path(args.expected_metrics))
        overall = next(row for row in rows if row["bucket"] == "overall")
        checks = {
            "bbox_AP": overall["bbox"]["AP"],
            "bbox_AP50": overall["bbox"]["AP50"],
            "bbox_AP75": overall["bbox"]["AP75"],
            "segm_AP": overall["segm"]["AP"],
            "segm_AP50": overall["segm"]["AP50"],
            "segm_AP75": overall["segm"]["AP75"],
        }
        for key, value in checks.items():
            diff = abs(float(expected[key]) - float(value))
            if diff > float(args.tolerance):
                raise DiagnosisError(f"overall metric mismatch for {key}: expected {expected[key]}, got {value}")

    if out_md:
        lines = [
            "| bucket | images | annotations | bbox AP/AP50/AP75 | segm AP/AP50/AP75 | mask oracle R@50/R@75 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            bbox = " / ".join(_format_float(row["bbox"][key]) for key in METRIC_KEYS)
            segm = " / ".join(_format_float(row["segm"][key]) for key in METRIC_KEYS)
            oracle = f"{_format_float(row['mask_oracle']['R50'])} / {_format_float(row['mask_oracle']['R75'])}"
            lines.append(f"| {row['bucket']} | {row['images']} | {row['annotations']} | {bbox} | {segm} | {oracle} |")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--r52-stats", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--expected-metrics", type=Path)
    parser.add_argument("--bottom-fraction", type=float, default=0.20)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps({"buckets": len(payload["buckets"]), "metadata": payload["metadata"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
