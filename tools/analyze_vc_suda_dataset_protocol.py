#!/usr/bin/env python3
"""Summarize COCO dataset protocol and prediction scale distributions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pycocotools import mask as mask_utils


DENSITY_BUCKETS = (
    ("0", 0, 0),
    ("1-24", 1, 24),
    ("25-30", 25, 30),
    ("31-45", 31, 45),
    ("46-60", 46, 60),
    ("61-90", 61, 90),
    (">90", 91, math.inf),
)


class ProtocolError(RuntimeError):
    """Raised when a protocol input is missing or malformed."""


def _load_json(path: str | Path) -> Any:
    resolved = Path(path)
    if not resolved.exists():
        raise ProtocolError(f"input file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{context} must be a JSON object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProtocolError(f"{context} must be a JSON list")
    return value


def _require_field(row: dict[str, Any], field: str, context: str) -> Any:
    if field not in row:
        raise ProtocolError(f"{context} missing required field: {field}")
    return row[field]


def _require_number(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ProtocolError(f"{context} must be a finite number")
    return float(value)


def _require_bbox(row: dict[str, Any], context: str) -> tuple[float, float, float, float]:
    bbox = _require_field(row, "bbox", context)
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ProtocolError(f"{context} bbox must be COCO [x, y, w, h]")
    x, y, width, height = (_require_number(value, f"{context} bbox[{idx}]") for idx, value in enumerate(bbox))
    if width < 0 or height < 0:
        raise ProtocolError(f"{context} bbox width/height must be non-negative")
    return x, y, width, height


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }

    def percentile(percent: float) -> float:
        if len(finite) == 1:
            return finite[0]
        rank = (len(finite) - 1) * percent / 100.0
        low = int(math.floor(rank))
        high = int(math.ceil(rank))
        if low == high:
            return finite[low]
        return finite[low] + (finite[high] - finite[low]) * (rank - low)

    return {
        "count": len(finite),
        "min": finite[0],
        "p10": percentile(10),
        "p25": percentile(25),
        "p50": percentile(50),
        "p75": percentile(75),
        "p90": percentile(90),
        "max": finite[-1],
        "mean": sum(finite) / len(finite),
    }


def _ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_float = float(denominator)
    if denominator_float == 0.0:
        return None
    return float(numerator) / denominator_float


def _bucket_name(count: int) -> str:
    for name, low, high in DENSITY_BUCKETS:
        if low <= count <= high:
            return name
    raise AssertionError(f"unreachable density bucket for count={count}")


def _segmentation_area(segmentation: Any, height: int, width: int, context: str) -> float:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        return float(mask_utils.area(mask_utils.merge(rles)))
    if isinstance(segmentation, dict) and "counts" in segmentation:
        return float(mask_utils.area(segmentation))
    raise ProtocolError(f"{context} segmentation must be COCO polygon or RLE")


def _parse_named_path(raw: str, option: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ProtocolError(f"{option} must use NAME=PATH")
    name, path = raw.split("=", 1)
    if not name:
        raise ProtocolError(f"{option} name is empty")
    if not path:
        raise ProtocolError(f"{option} path is empty")
    return name, Path(path)


def _parse_prediction(raw: str) -> tuple[str, str, Path]:
    parts = raw.split("=", 2)
    if len(parts) != 3:
        raise ProtocolError("--pred must use NAME=ANN_NAME=PATH")
    name, annotation_name, path = parts
    if not name:
        raise ProtocolError("--pred name is empty")
    if not annotation_name:
        raise ProtocolError("--pred annotation name is empty")
    if not path:
        raise ProtocolError("--pred path is empty")
    return name, annotation_name, Path(path)


def _summarize_coco(name: str, path: Path) -> dict[str, Any]:
    coco = _require_mapping(_load_json(path), f"annotation {name}")
    images = _require_list(_require_field(coco, "images", f"annotation {name}"), f"annotation {name}.images")
    annotations = _require_list(
        _require_field(coco, "annotations", f"annotation {name}"),
        f"annotation {name}.annotations",
    )
    _require_list(_require_field(coco, "categories", f"annotation {name}"), f"annotation {name}.categories")

    image_by_id: dict[int, dict[str, Any]] = {}
    counts_by_image: dict[int, int] = {}
    for image in images:
        image_row = _require_mapping(image, f"annotation {name} image")
        image_id = int(_require_field(image_row, "id", f"annotation {name} image"))
        _require_field(image_row, "file_name", f"annotation {name} image {image_id}")
        _require_number(_require_field(image_row, "width", f"annotation {name} image {image_id}"), f"annotation {name} image {image_id} width")
        _require_number(_require_field(image_row, "height", f"annotation {name} image {image_id}"), f"annotation {name} image {image_id} height")
        if image_id in image_by_id:
            raise ProtocolError(f"annotation {name} duplicate image id: {image_id}")
        image_by_id[image_id] = image_row
        counts_by_image[image_id] = 0

    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    bbox_areas: list[float] = []
    bbox_aspects: list[float] = []
    mask_areas: list[float] = []
    mask_area_over_bbox_area: list[float] = []

    for annotation in annotations:
        ann = _require_mapping(annotation, f"annotation {name} row")
        ann_id = int(_require_field(ann, "id", f"annotation {name} annotation"))
        image_id = int(_require_field(ann, "image_id", f"annotation {name} annotation {ann_id}"))
        if image_id not in image_by_id:
            raise ProtocolError(f"annotation {name} annotation {ann_id} references missing image_id {image_id}")
        _require_field(ann, "category_id", f"annotation {name} annotation {ann_id}")
        _, _, width, height = _require_bbox(ann, f"annotation {name} annotation {ann_id}")
        area = _require_number(_require_field(ann, "area", f"annotation {name} annotation {ann_id}"), f"annotation {name} annotation {ann_id} area")
        _require_field(ann, "segmentation", f"annotation {name} annotation {ann_id}")

        bbox_area = width * height
        bbox_widths.append(width)
        bbox_heights.append(height)
        bbox_areas.append(bbox_area)
        bbox_aspects.append(width / height if height else math.inf)
        mask_areas.append(area)
        mask_area_over_bbox_area.append(area / bbox_area if bbox_area else math.inf)
        counts_by_image[image_id] += 1

    density_buckets: dict[str, dict[str, int]] = {bucket[0]: {"images": 0, "instances": 0} for bucket in DENSITY_BUCKETS}
    for count in counts_by_image.values():
        bucket = density_buckets[_bucket_name(count)]
        bucket["images"] += 1
        bucket["instances"] += count

    return {
        "path": str(path),
        "images": len(images),
        "instances": len(annotations),
        "instances_per_image": _quantiles([float(count) for count in counts_by_image.values()]),
        "bbox_width": _quantiles(bbox_widths),
        "bbox_height": _quantiles(bbox_heights),
        "bbox_area": _quantiles(bbox_areas),
        "bbox_aspect_width_over_height": _quantiles(bbox_aspects),
        "mask_area": _quantiles(mask_areas),
        "mask_area_over_bbox_area": _quantiles(mask_area_over_bbox_area),
        "area_le_256_ratio": (sum(1 for area in mask_areas if area <= 256.0) / len(mask_areas)) if mask_areas else None,
        "density_buckets": density_buckets,
        "_image_by_id": image_by_id,
        "_counts_by_image": counts_by_image,
    }


def _summarize_predictions(name: str, annotation_name: str, path: Path, dataset: dict[str, Any]) -> dict[str, Any]:
    predictions = _require_list(_load_json(path), f"prediction {name}")
    image_by_id = dataset["_image_by_id"]
    gt_counts_by_image = dataset["_counts_by_image"]
    pred_counts_by_image: dict[int, int] = {int(image_id): 0 for image_id in image_by_id}

    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    bbox_areas: list[float] = []
    bbox_aspects: list[float] = []
    mask_areas: list[float] = []
    mask_area_over_bbox_area: list[float] = []
    scores: list[float] = []

    for idx, prediction in enumerate(predictions):
        pred = _require_mapping(prediction, f"prediction {name} row {idx}")
        image_id = int(_require_field(pred, "image_id", f"prediction {name} row {idx}"))
        if image_id not in image_by_id:
            raise ProtocolError(f"prediction {name} row {idx} references image_id {image_id} outside annotation {annotation_name}")
        _require_field(pred, "category_id", f"prediction {name} row {idx}")
        score = _require_number(_require_field(pred, "score", f"prediction {name} row {idx}"), f"prediction {name} row {idx} score")
        _, _, width, height = _require_bbox(pred, f"prediction {name} row {idx}")
        image = image_by_id[image_id]
        image_height = int(_require_number(image["height"], f"annotation {annotation_name} image {image_id} height"))
        image_width = int(_require_number(image["width"], f"annotation {annotation_name} image {image_id} width"))
        mask_area = _segmentation_area(
            _require_field(pred, "segmentation", f"prediction {name} row {idx}"),
            image_height,
            image_width,
            f"prediction {name} row {idx}",
        )

        bbox_area = width * height
        bbox_widths.append(width)
        bbox_heights.append(height)
        bbox_areas.append(bbox_area)
        bbox_aspects.append(width / height if height else math.inf)
        mask_areas.append(mask_area)
        mask_area_over_bbox_area.append(mask_area / bbox_area if bbox_area else math.inf)
        scores.append(score)
        pred_counts_by_image[image_id] += 1

    pred_count_quantiles = _quantiles([float(count) for count in pred_counts_by_image.values()])
    gt_count_quantiles = dataset["instances_per_image"]
    pred_mask_area = _quantiles(mask_areas)
    pred_bbox_width = _quantiles(bbox_widths)
    pred_bbox_height = _quantiles(bbox_heights)
    gt_mask_area = dataset["mask_area"]
    gt_bbox_width = dataset["bbox_width"]
    gt_bbox_height = dataset["bbox_height"]

    pred_per_gt = [
        (float(pred_counts_by_image[image_id]) / float(gt_count))
        for image_id, gt_count in gt_counts_by_image.items()
        if gt_count > 0
    ]

    return {
        "path": str(path),
        "annotation_name": annotation_name,
        "predictions": len(predictions),
        "prediction_count_per_image": pred_count_quantiles,
        "predictions_per_gt_instance": _quantiles(pred_per_gt),
        "score": _quantiles(scores),
        "bbox_width": pred_bbox_width,
        "bbox_height": pred_bbox_height,
        "bbox_area": _quantiles(bbox_areas),
        "bbox_aspect_width_over_height": _quantiles(bbox_aspects),
        "mask_area": pred_mask_area,
        "mask_area_over_bbox_area": _quantiles(mask_area_over_bbox_area),
        "scale_vs_gt": {
            "prediction_count_per_image_p50_delta": (
                float(pred_count_quantiles["p50"]) - float(gt_count_quantiles["p50"])
                if pred_count_quantiles["p50"] is not None and gt_count_quantiles["p50"] is not None
                else None
            ),
            "prediction_count_per_image_p50_ratio": _ratio(pred_count_quantiles["p50"], gt_count_quantiles["p50"]),
            "mask_area_p50_ratio": _ratio(pred_mask_area["p50"], gt_mask_area["p50"]),
            "mask_area_p10_ratio": _ratio(pred_mask_area["p10"], gt_mask_area["p10"]),
            "mask_area_p90_ratio": _ratio(pred_mask_area["p90"], gt_mask_area["p90"]),
            "bbox_width_p50_ratio": _ratio(pred_bbox_width["p50"], gt_bbox_width["p50"]),
            "bbox_height_p50_ratio": _ratio(pred_bbox_height["p50"], gt_bbox_height["p50"]),
        },
    }


def _drop_private(summary: dict[str, Any]) -> dict[str, Any]:
    public = {"datasets": {}, "predictions": summary["predictions"]}
    for name, dataset in summary["datasets"].items():
        public["datasets"][name] = {key: value for key, value in dataset.items() if not key.startswith("_")}
    return public


def _markdown_table(summary: dict[str, Any]) -> str:
    lines = [
        "| name | images | instances | inst/img p50 | mask area p50 | area<=256 | density 25-30 | density 46-60 | density >90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, dataset in summary["datasets"].items():
        area_ratio = dataset["area_le_256_ratio"]
        lines.append(
            "| {name} | {images} | {instances} | {p50:.3f} | {area:.3f} | {small:.6f} | {b25} | {b46} | {b90} |".format(
                name=name,
                images=dataset["images"],
                instances=dataset["instances"],
                p50=float(dataset["instances_per_image"]["p50"] or 0.0),
                area=float(dataset["mask_area"]["p50"] or 0.0),
                small=float(area_ratio) if area_ratio is not None else 0.0,
                b25=dataset["density_buckets"]["25-30"]["images"],
                b46=dataset["density_buckets"]["46-60"]["images"],
                b90=dataset["density_buckets"][">90"]["images"],
            )
        )

    if summary["predictions"]:
        lines.extend(
            [
                "",
                "| prediction | annotation | predictions | pred/img p50 | pred/GT img p50 ratio | pred mask/GT p50 ratio |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for name, prediction in summary["predictions"].items():
            scale = prediction["scale_vs_gt"]
            lines.append(
                "| {name} | {ann} | {preds} | {p50:.3f} | {cnt_ratio:.6f} | {area_ratio:.6f} |".format(
                    name=name,
                    ann=prediction["annotation_name"],
                    preds=prediction["predictions"],
                    p50=float(prediction["prediction_count_per_image"]["p50"] or 0.0),
                    cnt_ratio=float(scale["prediction_count_per_image_p50_ratio"] or 0.0),
                    area_ratio=float(scale["mask_area_p50_ratio"] or 0.0),
                )
            )
    return "\n".join(lines) + "\n"


def run_analysis(
    *,
    annotations: dict[str, Path],
    predictions: dict[str, tuple[str, Path]],
) -> dict[str, Any]:
    datasets: dict[str, dict[str, Any]] = {}
    for name, path in annotations.items():
        if name in datasets:
            raise ProtocolError(f"duplicate annotation name: {name}")
        datasets[name] = _summarize_coco(name, path)

    prediction_summaries: dict[str, dict[str, Any]] = {}
    for name, (annotation_name, path) in predictions.items():
        if annotation_name not in datasets:
            raise ProtocolError(f"prediction {name} references unknown annotation name: {annotation_name}")
        prediction_summaries[name] = _summarize_predictions(name, annotation_name, path, datasets[annotation_name])

    return _drop_private({"datasets": datasets, "predictions": prediction_summaries})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", action="append", default=[], help="COCO annotation input as NAME=PATH. Repeatable.")
    parser.add_argument("--pred", action="append", default=[], help="Prediction input as NAME=ANN_NAME=PATH. Repeatable.")
    parser.add_argument("--output-json", required=True, help="Path to write JSON summary.")
    parser.add_argument("--output-md", help="Optional path to write markdown-friendly tables.")
    args = parser.parse_args(argv)

    annotations: dict[str, Path] = {}
    for raw in args.ann:
        name, path = _parse_named_path(raw, "--ann")
        if name in annotations:
            raise ProtocolError(f"duplicate annotation name: {name}")
        annotations[name] = path
    if not annotations:
        raise ProtocolError("at least one --ann NAME=PATH is required")

    predictions: dict[str, tuple[str, Path]] = {}
    for raw in args.pred:
        name, annotation_name, path = _parse_prediction(raw)
        if name in predictions:
            raise ProtocolError(f"duplicate prediction name: {name}")
        predictions[name] = (annotation_name, path)

    summary = run_analysis(annotations=annotations, predictions=predictions)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_markdown_table(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
