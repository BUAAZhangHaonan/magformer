#!/usr/bin/env python3
"""Build a target-scale image-level subset from a COCO source annotation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


DENSITY_BUCKETS = (
    ("0", 0, 0),
    ("1-24", 1, 24),
    ("25-30", 25, 30),
    ("31-45", 31, 45),
    ("46-60", 46, 60),
    ("61-90", 61, 90),
    (">90", 91, math.inf),
)

SCORING_WEIGHTS = {
    "image_median_area": 0.5,
    "small_area_ratio": 0.3,
    "instance_density": 0.2,
}


class SubsetBuildError(RuntimeError):
    """Raised when a subset builder input is missing or malformed."""


def _load_json(path: str | Path) -> Any:
    resolved = Path(path)
    if not resolved.exists():
        raise SubsetBuildError(f"input file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SubsetBuildError(f"{context} must be a JSON object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise SubsetBuildError(f"{context} must be a JSON list")
    return value


def _require_field(row: dict[str, Any], field: str, context: str) -> Any:
    if field not in row:
        raise SubsetBuildError(f"{context} missing required field: {field}")
    return row[field]


def _require_number(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SubsetBuildError(f"{context} must be a finite number")
    return float(value)


def _require_id(value: Any, context: str) -> int:
    if not isinstance(value, int):
        raise SubsetBuildError(f"{context} must be an integer id")
    return value


def _require_bbox(row: dict[str, Any], context: str) -> tuple[float, float, float, float]:
    bbox = _require_field(row, "bbox", context)
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise SubsetBuildError(f"{context} bbox must be COCO [x, y, w, h]")
    x, y, width, height = (_require_number(value, f"{context} bbox[{idx}]") for idx, value in enumerate(bbox))
    if width <= 0 or height <= 0:
        raise SubsetBuildError(f"{context} bbox width/height must be positive")
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


def _bucket_name(count: int) -> str:
    for name, low, high in DENSITY_BUCKETS:
        if low <= count <= high:
            return name
    raise AssertionError(f"unreachable density bucket for count={count}")


def _median(values: list[float]) -> float:
    if not values:
        raise SubsetBuildError("cannot compute median for empty value list")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _ratio_or_none(numerator: float | int, denominator: float | int) -> float | None:
    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _relative_distance(value: float, target: float) -> float:
    if value <= 0.0 or target <= 0.0:
        raise SubsetBuildError("scale distance requires positive values")
    return abs(math.log(value / target))


def _load_coco(path: Path, name: str, area_key: str) -> dict[str, Any]:
    coco = _require_mapping(_load_json(path), f"{name} annotation")
    images = _require_list(_require_field(coco, "images", f"{name} annotation"), f"{name}.images")
    annotations = _require_list(
        _require_field(coco, "annotations", f"{name} annotation"),
        f"{name}.annotations",
    )
    categories = _require_list(_require_field(coco, "categories", f"{name} annotation"), f"{name}.categories")

    image_by_id: dict[int, dict[str, Any]] = {}
    for idx, image in enumerate(images):
        row = _require_mapping(image, f"{name} image row {idx}")
        image_id = _require_id(_require_field(row, "id", f"{name} image row {idx}"), f"{name} image row {idx} id")
        _require_field(row, "file_name", f"{name} image {image_id}")
        width = _require_number(_require_field(row, "width", f"{name} image {image_id}"), f"{name} image {image_id} width")
        height = _require_number(_require_field(row, "height", f"{name} image {image_id}"), f"{name} image {image_id} height")
        if width <= 0 or height <= 0:
            raise SubsetBuildError(f"{name} image {image_id} width/height must be positive")
        if image_id in image_by_id:
            raise SubsetBuildError(f"{name} duplicate image id: {image_id}")
        image_by_id[image_id] = row

    category_ids: set[int] = set()
    for idx, category in enumerate(categories):
        row = _require_mapping(category, f"{name} category row {idx}")
        category_id = _require_id(
            _require_field(row, "id", f"{name} category row {idx}"),
            f"{name} category row {idx} id",
        )
        if category_id in category_ids:
            raise SubsetBuildError(f"{name} duplicate category id: {category_id}")
        category_ids.add(category_id)
    if not category_ids:
        raise SubsetBuildError(f"{name} categories must not be empty")

    ann_by_id: dict[int, dict[str, Any]] = {}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_by_id}
    for idx, annotation in enumerate(annotations):
        row = _require_mapping(annotation, f"{name} annotation row {idx}")
        ann_id = _require_id(_require_field(row, "id", f"{name} annotation row {idx}"), f"{name} annotation row {idx} id")
        if ann_id in ann_by_id:
            raise SubsetBuildError(f"{name} duplicate annotation id: {ann_id}")
        image_id = _require_id(_require_field(row, "image_id", f"{name} annotation {ann_id}"), f"{name} annotation {ann_id} image_id")
        if image_id not in image_by_id:
            raise SubsetBuildError(f"{name} annotation {ann_id} references missing image_id {image_id}")
        category_id = _require_id(
            _require_field(row, "category_id", f"{name} annotation {ann_id}"),
            f"{name} annotation {ann_id} category_id",
        )
        if category_id not in category_ids:
            raise SubsetBuildError(f"{name} annotation {ann_id} has invalid category_id {category_id}")
        _require_bbox(row, f"{name} annotation {ann_id}")
        area = _require_number(_require_field(row, area_key, f"{name} annotation {ann_id}"), f"{name} annotation {ann_id} {area_key}")
        if area <= 0:
            raise SubsetBuildError(f"{name} annotation {ann_id} {area_key} must be positive")
        ann_by_id[ann_id] = row
        annotations_by_image[image_id].append(row)

    return {
        "path": str(path),
        "raw": coco,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "image_by_id": image_by_id,
        "annotations_by_image": annotations_by_image,
    }


def _summarize_coco(coco: dict[str, Any], *, area_key: str) -> dict[str, Any]:
    images = coco["images"]
    annotations_by_image = coco["annotations_by_image"]
    instance_counts = [len(annotations_by_image[int(image["id"])]) for image in images]
    areas = [float(annotation[area_key]) for annotation in coco["annotations"]]
    density_buckets: dict[str, dict[str, int]] = {bucket[0]: {"images": 0, "instances": 0} for bucket in DENSITY_BUCKETS}
    for count in instance_counts:
        bucket = density_buckets[_bucket_name(count)]
        bucket["images"] += 1
        bucket["instances"] += count
    return {
        "path": coco["path"],
        "images": len(images),
        "instances": len(coco["annotations"]),
        "instances_per_image": _quantiles([float(count) for count in instance_counts]),
        "mask_area": _quantiles(areas),
        "area_le_256_ratio": (sum(1 for area in areas if area <= 256.0) / len(areas)) if areas else None,
        "density_buckets": density_buckets,
    }


def _image_profile(image: dict[str, Any], annotations: list[dict[str, Any]], *, area_key: str) -> dict[str, Any]:
    image_id = int(image["id"])
    areas = [float(annotation[area_key]) for annotation in annotations]
    if not areas:
        raise SubsetBuildError(f"source image {image_id} has zero annotations; cannot score target-scale subset")
    return {
        "image_id": image_id,
        "instance_count": len(annotations),
        "median_area": _median(areas),
        "small_area_ratio": sum(1 for area in areas if area <= 256.0) / len(areas),
    }


def _score_profile(profile: dict[str, Any], target: dict[str, float]) -> dict[str, float]:
    median_distance = _relative_distance(float(profile["median_area"]), target["median_area"])
    small_distance = abs(float(profile["small_area_ratio"]) - target["small_area_ratio"])
    density_distance = abs(float(profile["instance_count"]) - target["density_p50"]) / max(target["density_p50"], 1.0)
    total = (
        SCORING_WEIGHTS["image_median_area"] * median_distance
        + SCORING_WEIGHTS["small_area_ratio"] * small_distance
        + SCORING_WEIGHTS["instance_density"] * density_distance
    )
    return {
        "total": total,
        "image_median_area": median_distance,
        "small_area_ratio": small_distance,
        "instance_density": density_distance,
    }


def _build_output_coco(source: dict[str, Any], selected_ids: set[int]) -> dict[str, Any]:
    source_raw = source["raw"]
    output: dict[str, Any] = {}
    if "info" in source_raw:
        output["info"] = source_raw["info"]
    if "licenses" in source_raw:
        output["licenses"] = source_raw["licenses"]
    output["images"] = [image for image in source["images"] if int(image["id"]) in selected_ids]
    output["annotations"] = [ann for ann in source["annotations"] if int(ann["image_id"]) in selected_ids]
    output["categories"] = source["categories"]
    return output


def _summary_for_output(output: dict[str, Any], output_ann: Path) -> dict[str, Any]:
    annotations_by_image: dict[int, list[dict[str, Any]]] = {int(image["id"]): [] for image in output["images"]}
    for annotation in output["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    return {
        "path": str(output_ann),
        "raw": output,
        "images": output["images"],
        "annotations": output["annotations"],
        "categories": output["categories"],
        "annotations_by_image": annotations_by_image,
    }


def _resolve_target_images(target_images: int | None, max_images: int | None) -> int:
    if target_images is not None and max_images is not None:
        raise SubsetBuildError("use only one of --target-images or --max-images")
    selected = target_images if target_images is not None else max_images
    if selected is None:
        selected = 200
    if selected <= 0:
        raise SubsetBuildError("target image count must be positive")
    return selected


def build_subset(
    *,
    source_ann: str | Path,
    target_ann: str | Path,
    output_ann: str | Path,
    summary_json: str | Path,
    target_images: int | None = None,
    max_images: int | None = None,
    seed: int = 0,
    area_key: str = "area",
) -> dict[str, Any]:
    source_path = Path(source_ann)
    target_path = Path(target_ann)
    output_path = Path(output_ann)
    summary_path = Path(summary_json)
    image_count = _resolve_target_images(target_images, max_images)

    source = _load_coco(source_path, "source", area_key)
    target = _load_coco(target_path, "target", area_key)
    if image_count > len(source["images"]):
        raise SubsetBuildError(f"requested {image_count} images, but source has only {len(source['images'])}")

    target_summary = _summarize_coco(target, area_key=area_key)
    target_median_area = target_summary["mask_area"]["p50"]
    target_small_ratio = target_summary["area_le_256_ratio"]
    target_density_p50 = target_summary["instances_per_image"]["p50"]
    if target_median_area is None or target_small_ratio is None or target_density_p50 is None:
        raise SubsetBuildError("target annotation must contain at least one annotation")

    target_profile = {
        "median_area": float(target_median_area),
        "small_area_ratio": float(target_small_ratio),
        "density_p50": float(target_density_p50),
    }

    rng = random.Random(seed)
    scored: list[dict[str, Any]] = []
    for image in source["images"]:
        image_id = int(image["id"])
        profile = _image_profile(image, source["annotations_by_image"][image_id], area_key=area_key)
        score = _score_profile(profile, target_profile)
        scored.append({"image_id": image_id, "profile": profile, "score": score, "tie_breaker": rng.random()})
    scored.sort(key=lambda item: (float(item["score"]["total"]), float(item["tie_breaker"]), int(item["image_id"])))

    selected = scored[:image_count]
    selected_ids = {int(item["image_id"]) for item in selected}
    output = _build_output_coco(source, selected_ids)
    output_summary_obj = _summary_for_output(output, output_path)

    source_summary = _summarize_coco(source, area_key=area_key)
    subset_summary = _summarize_coco(output_summary_obj, area_key=area_key)
    subset_summary["selection"] = {
        "target_images": image_count,
        "seed": seed,
        "area_key": area_key,
        "scoring_weights": SCORING_WEIGHTS,
        "score_components": "weighted sum of image median-area log distance, small-area-ratio absolute distance, and instance-density relative distance",
        "selected_image_ids": [int(item["image_id"]) for item in selected],
        "selected_score_quantiles": _quantiles([float(item["score"]["total"]) for item in selected]),
        "source_to_subset_mask_area_p50_ratio": _ratio_or_none(
            float(subset_summary["mask_area"]["p50"]),
            float(source_summary["mask_area"]["p50"]),
        )
        if subset_summary["mask_area"]["p50"] is not None and source_summary["mask_area"]["p50"] is not None
        else None,
        "subset_to_target_mask_area_p50_ratio": _ratio_or_none(
            float(subset_summary["mask_area"]["p50"]),
            float(target_summary["mask_area"]["p50"]),
        )
        if subset_summary["mask_area"]["p50"] is not None and target_summary["mask_area"]["p50"] is not None
        else None,
    }

    summary = {
        "source": source_summary,
        "target": target_summary,
        "subset": subset_summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ann", required=True, help="COCO source annotation path.")
    parser.add_argument("--target-ann", required=True, help="COCO target annotation path.")
    parser.add_argument("--output-ann", required=True, help="Path to write the COCO image-level source subset.")
    parser.add_argument("--summary-json", required=True, help="Path to write target/source/subset distribution summary JSON.")
    parser.add_argument("--target-images", type=int, help="Number of source images to select. Defaults to 200.")
    parser.add_argument("--max-images", type=int, help="Alias for --target-images. Use only one of the two options.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic tie-break seed.")
    parser.add_argument("--area-key", default="area", help="Annotation area field to use. Defaults to COCO 'area'.")
    args = parser.parse_args(argv)

    try:
        summary = build_subset(
            source_ann=args.source_ann,
            target_ann=args.target_ann,
            output_ann=args.output_ann,
            summary_json=args.summary_json,
            target_images=args.target_images,
            max_images=args.max_images,
            seed=args.seed,
            area_key=args.area_key,
        )
    except SubsetBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
