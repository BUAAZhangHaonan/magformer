#!/usr/bin/env python3
"""Diagnose whether source annotations can form a target-scale pool."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


class DiagnosisError(RuntimeError):
    """Raised when an input is missing or malformed."""


def _load_json(path: str | Path) -> Any:
    resolved = Path(path)
    if not resolved.exists():
        raise DiagnosisError(f"input file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DiagnosisError(f"{context} must be a JSON object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise DiagnosisError(f"{context} must be a JSON list")
    return value


def _require_field(row: dict[str, Any], field: str, context: str) -> Any:
    if field not in row:
        raise DiagnosisError(f"{context} missing required field: {field}")
    return row[field]


def _require_number(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DiagnosisError(f"{context} must be a finite number")
    return float(value)


def _require_id(value: Any, context: str) -> int:
    if not isinstance(value, int):
        raise DiagnosisError(f"{context} must be an integer id")
    return value


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


def _parse_named_path(raw: str, option: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise DiagnosisError(f"{option} must use NAME=PATH")
    name, path = raw.split("=", 1)
    if not name:
        raise DiagnosisError(f"{option} name is empty")
    if not path:
        raise DiagnosisError(f"{option} path is empty")
    return name, Path(path)


def _basename(file_name: Any, context: str) -> str:
    if not isinstance(file_name, str) or not file_name:
        raise DiagnosisError(f"{context} file_name must be a non-empty string")
    return Path(file_name).name


def _load_coco(path: Path, name: str) -> dict[str, Any]:
    coco = _require_mapping(_load_json(path), f"{name} annotation")
    images = _require_list(_require_field(coco, "images", f"{name} annotation"), f"{name}.images")
    annotations = _require_list(
        _require_field(coco, "annotations", f"{name} annotation"),
        f"{name}.annotations",
    )
    categories = _require_list(_require_field(coco, "categories", f"{name} annotation"), f"{name}.categories")
    if not categories:
        raise DiagnosisError(f"{name}.categories must not be empty")

    image_by_id: dict[int, dict[str, Any]] = {}
    counts_by_image: dict[int, int] = {}
    basenames: list[str] = []
    for idx, image in enumerate(images):
        row = _require_mapping(image, f"{name} image row {idx}")
        image_id = _require_id(_require_field(row, "id", f"{name} image row {idx}"), f"{name} image row {idx} id")
        if image_id in image_by_id:
            raise DiagnosisError(f"{name} duplicate image id: {image_id}")
        width = _require_number(_require_field(row, "width", f"{name} image {image_id}"), f"{name} image {image_id} width")
        height = _require_number(_require_field(row, "height", f"{name} image {image_id}"), f"{name} image {image_id} height")
        if width <= 0 or height <= 0:
            raise DiagnosisError(f"{name} image {image_id} width/height must be positive")
        basenames.append(_basename(_require_field(row, "file_name", f"{name} image {image_id}"), f"{name} image {image_id}"))
        image_by_id[image_id] = row
        counts_by_image[image_id] = 0

    areas: list[float] = []
    ann_ids: set[int] = set()
    for idx, annotation in enumerate(annotations):
        row = _require_mapping(annotation, f"{name} annotation row {idx}")
        ann_id = _require_id(_require_field(row, "id", f"{name} annotation row {idx}"), f"{name} annotation row {idx} id")
        if ann_id in ann_ids:
            raise DiagnosisError(f"{name} duplicate annotation id: {ann_id}")
        ann_ids.add(ann_id)
        image_id = _require_id(_require_field(row, "image_id", f"{name} annotation {ann_id}"), f"{name} annotation {ann_id} image_id")
        if image_id not in image_by_id:
            raise DiagnosisError(f"{name} annotation {ann_id} references missing image_id {image_id}")
        area = _require_number(_require_field(row, "area", f"{name} annotation {ann_id}"), f"{name} annotation {ann_id} area")
        if area <= 0:
            raise DiagnosisError(f"{name} annotation {ann_id} area must be positive")
        _require_field(row, "bbox", f"{name} annotation {ann_id}")
        _require_field(row, "segmentation", f"{name} annotation {ann_id}")
        areas.append(area)
        counts_by_image[image_id] += 1

    return {
        "name": name,
        "path": str(path),
        "images": images,
        "annotations": annotations,
        "areas": areas,
        "instance_counts": [counts_by_image[int(image["id"])] for image in images],
        "basenames": basenames,
    }


def _overlap_summary(candidate_basenames: list[str], forbidden: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate_set = set(candidate_basenames)
    result: dict[str, Any] = {}
    total = 0
    for name, coco in forbidden.items():
        overlap = sorted(candidate_set & set(coco["basenames"]))
        total += len(overlap)
        result[name] = {
            "count": len(overlap),
            "examples": overlap[:10],
        }
    result["total"] = total
    return result


def _duplicate_basename_count(basenames: list[str]) -> int:
    return len(basenames) - len(set(basenames))


def _summarize_distribution(
    *,
    name: str,
    path: str | None,
    areas: list[float],
    instance_counts: list[int],
    basenames: list[str],
    forbidden: dict[str, dict[str, Any]],
    gate_config: dict[str, float | int],
) -> dict[str, Any]:
    if not areas:
        raise DiagnosisError(f"{name} must contain at least one annotation")
    area_summary = _quantiles(areas)
    count_summary = _quantiles([float(count) for count in instance_counts])
    overlap = _overlap_summary(basenames, forbidden)
    p50 = area_summary["p50"]
    if p50 is None:
        raise DiagnosisError(f"{name} mask area p50 is undefined")
    area_le_256_ratio = sum(1 for area in areas if area <= 256.0) / len(areas)
    small_target = float(gate_config["target_area_le_256_ratio"])
    small_delta = abs(area_le_256_ratio - small_target)
    gate = {
        "min_images_pass": len(instance_counts) >= int(gate_config["min_images"]),
        "mask_area_p50_pass": float(gate_config["p50_lower"]) <= float(p50) <= float(gate_config["p50_upper"]),
        "area_le_256_ratio_pass": small_delta <= float(gate_config["small_area_abs_tol"]),
        "basename_overlap_pass": int(overlap["total"]) == 0,
        "area_le_256_ratio_abs_delta": small_delta,
    }
    gate["pass"] = all(
        bool(gate[key])
        for key in ("min_images_pass", "mask_area_p50_pass", "area_le_256_ratio_pass", "basename_overlap_pass")
    )
    return {
        "name": name,
        "path": path,
        "images": len(instance_counts),
        "instances": len(areas),
        "instances_per_image": count_summary,
        "mask_area": area_summary,
        "area_le_256_ratio": area_le_256_ratio,
        "basename_overlap": overlap,
        "duplicate_basenames": _duplicate_basename_count(basenames),
        "gate": gate,
    }


def _base_summary(coco: dict[str, Any], forbidden: dict[str, dict[str, Any]], gate_config: dict[str, float | int]) -> dict[str, Any]:
    return _summarize_distribution(
        name=str(coco["name"]),
        path=str(coco["path"]),
        areas=[float(area) for area in coco["areas"]],
        instance_counts=[int(count) for count in coco["instance_counts"]],
        basenames=[str(name) for name in coco["basenames"]],
        forbidden=forbidden,
        gate_config=gate_config,
    )


def diagnose_pool(
    *,
    source_ann: str | Path,
    target_ann: str | Path,
    scale_factors: list[float],
    summary_json: str | Path,
    existing_pool_ann: str | Path | None = None,
    forbidden_anns: dict[str, str | Path] | None = None,
    min_images: int = 4000,
    p50_lower: float | None = None,
    p50_upper: float | None = None,
    small_area_abs_tol: float = 0.05,
) -> dict[str, Any]:
    if not scale_factors:
        raise DiagnosisError("at least one scale factor is required")
    if min_images <= 0:
        raise DiagnosisError("min_images must be positive")
    if small_area_abs_tol < 0.0:
        raise DiagnosisError("small_area_abs_tol must be non-negative")
    for scale in scale_factors:
        if scale <= 0.0 or not math.isfinite(scale):
            raise DiagnosisError(f"scale factor must be positive and finite: {scale}")

    source = _load_coco(Path(source_ann), "source")
    target = _load_coco(Path(target_ann), "target")
    existing = _load_coco(Path(existing_pool_ann), "existing_pool") if existing_pool_ann else None
    forbidden = {
        name: _load_coco(Path(path), name)
        for name, path in (forbidden_anns or {}).items()
    }

    target_area_summary = _quantiles([float(area) for area in target["areas"]])
    target_p50 = target_area_summary["p50"]
    if target_p50 is None:
        raise DiagnosisError("target annotation must contain at least one annotation")
    target_small_ratio = sum(1 for area in target["areas"] if float(area) <= 256.0) / len(target["areas"])
    lower = float(p50_lower) if p50_lower is not None else float(target_p50) * 0.75
    upper = float(p50_upper) if p50_upper is not None else float(target_p50) * 1.5
    if lower <= 0 or upper <= 0 or lower > upper:
        raise DiagnosisError("p50 gate bounds must be positive and lower <= upper")

    gate_config: dict[str, float | int] = {
        "min_images": int(min_images),
        "p50_lower": lower,
        "p50_upper": upper,
        "target_area_le_256_ratio": float(target_small_ratio),
        "small_area_abs_tol": float(small_area_abs_tol),
    }

    target_summary = _base_summary(target, forbidden, gate_config)
    existing_summary = _base_summary(existing, forbidden, gate_config) if existing else None

    source_areas = [float(area) for area in source["areas"]]
    source_counts = [int(count) for count in source["instance_counts"]]
    source_basenames = [str(name) for name in source["basenames"]]
    candidates: list[dict[str, Any]] = []
    for scale in scale_factors:
        scale_sq = float(scale) * float(scale)
        projected_areas = [area * scale_sq for area in source_areas]
        projected = _summarize_distribution(
            name=f"source_projected_s{scale:g}",
            path=str(source["path"]),
            areas=projected_areas,
            instance_counts=source_counts,
            basenames=source_basenames,
            forbidden=forbidden,
            gate_config=gate_config,
        )
        candidate: dict[str, Any] = {
            "scale_factor": float(scale),
            "area_scale_factor": scale_sq,
            "projected_source": projected,
        }
        if existing is not None:
            mixed = _summarize_distribution(
                name=f"mixed_existing_plus_source_projected_s{scale:g}",
                path=None,
                areas=[float(area) for area in existing["areas"]] + projected_areas,
                instance_counts=[int(count) for count in existing["instance_counts"]] + source_counts,
                basenames=[str(name) for name in existing["basenames"]] + source_basenames,
                forbidden=forbidden,
                gate_config=gate_config,
            )
            candidate["mixed_pool"] = mixed
        candidates.append(candidate)

    summary = {
        "schema_version": 1,
        "method": "annotation_area_projection_only_no_image_generation",
        "gate_config": gate_config,
        "source": _base_summary(source, forbidden, gate_config),
        "target": target_summary,
        "existing_pool": existing_summary,
        "candidates": candidates,
    }
    summary_path = Path(summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ann", required=True, help="COCO source annotation path.")
    parser.add_argument("--target-ann", required=True, help="COCO target annotation path.")
    parser.add_argument(
        "--scale-factors",
        nargs="+",
        type=float,
        required=True,
        help="Candidate linear resize scale factors, e.g. 0.20 0.24 0.25.",
    )
    parser.add_argument("--existing-pool-ann", help="Optional existing target-scale source pool annotation.")
    parser.add_argument(
        "--forbidden-ann",
        action="append",
        default=[],
        help="Forbidden overlap annotation as NAME=PATH. Repeat for target_labeled/target_unlabeled/val.",
    )
    parser.add_argument("--summary-json", required=True, help="Path to write JSON summary.")
    parser.add_argument("--min-images", type=int, default=4000, help="Minimum images required by gate.")
    parser.add_argument("--p50-lower", type=float, help="Absolute lower mask-area p50 gate. Defaults to 0.75 * target p50.")
    parser.add_argument("--p50-upper", type=float, help="Absolute upper mask-area p50 gate. Defaults to 1.5 * target p50.")
    parser.add_argument(
        "--small-area-abs-tol",
        type=float,
        default=0.05,
        help="Absolute tolerance for area<=256 ratio versus target. Defaults to 0.05.",
    )
    args = parser.parse_args(argv)

    try:
        forbidden_anns: dict[str, Path] = {}
        for raw in args.forbidden_ann:
            name, path = _parse_named_path(raw, "--forbidden-ann")
            if name in forbidden_anns:
                raise DiagnosisError(f"duplicate forbidden annotation name: {name}")
            forbidden_anns[name] = path
        summary = diagnose_pool(
            source_ann=args.source_ann,
            target_ann=args.target_ann,
            scale_factors=args.scale_factors,
            summary_json=args.summary_json,
            existing_pool_ann=args.existing_pool_ann,
            forbidden_anns=forbidden_anns,
            min_images=args.min_images,
            p50_lower=args.p50_lower,
            p50_upper=args.p50_upper,
            small_area_abs_tol=args.small_area_abs_tol,
        )
    except DiagnosisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
