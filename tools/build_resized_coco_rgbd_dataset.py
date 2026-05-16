#!/usr/bin/env python3
"""Build a small or full resized COCO RGB-D dataset from real RGB/depth assets."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


class BuildError(RuntimeError):
    """Raised when input data or output state violates the build contract."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BuildError(f"Input annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise BuildError(f"COCO annotation must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _require_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise BuildError(f"COCO annotation missing required list field: {key}")
    return value


def _require_number(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise BuildError(f"{context} must be a finite number")
    return float(value)


def _require_int(value: Any, context: str) -> int:
    if not isinstance(value, int):
        raise BuildError(f"{context} must be an integer")
    return int(value)


def _resolve_input_ann(source_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (source_root / path).resolve()


def _resolve_output_ann(output_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (output_root / path).resolve()


def _prepare_outputs(output_root: Path, output_ann: Path, summary_json: Path, overwrite: bool) -> None:
    existing = [path for path in (output_root, output_ann, summary_json) if path.exists()]
    if existing and not overwrite:
        raise BuildError("Output path already exists; pass --overwrite to replace: " + ", ".join(str(path) for path in existing))
    if overwrite:
        if output_root.exists():
            shutil.rmtree(output_root)
        for path in (output_ann, summary_json):
            if path.exists() and not path.is_relative_to(output_root):
                path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)


def _scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _resize_rgb(src_path: Path, dst_path: Path, width: int, height: int) -> None:
    if not src_path.exists():
        raise BuildError(f"RGB file not found: {src_path}")
    with Image.open(src_path) as image:
        rgb = image.convert("RGB")
        resized = rgb.resize((width, height), Image.Resampling.BILINEAR)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(dst_path)


def _resize_depth(src_path: Path, dst_path: Path, width: int, height: int) -> None:
    if not src_path.exists():
        raise BuildError(f"Depth file not found: {src_path}")
    depth = np.load(src_path, allow_pickle=False)
    if isinstance(depth, np.lib.npyio.NpzFile):
        raise BuildError(f"Depth file must be .npy, got npz payload: {src_path}")
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise BuildError(f"Depth file must be 2D or HxWx1, got shape {depth.shape}: {src_path}")
    resized = cv2.resize(depth.astype(np.float32, copy=False), (width, height), interpolation=cv2.INTER_LINEAR)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst_path, resized.astype(np.float32, copy=False))


def _decode_rle(segmentation: dict[str, Any], height: int, width: int, context: str) -> np.ndarray:
    if "size" not in segmentation or "counts" not in segmentation:
        raise BuildError(f"{context} RLE segmentation must contain size and counts")
    size = segmentation["size"]
    if size != [height, width]:
        raise BuildError(f"{context} RLE size {size!r} does not match image size {[height, width]}")
    counts = segmentation["counts"]
    if isinstance(counts, list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    elif isinstance(counts, str):
        rle = dict(segmentation)
        rle["counts"] = counts.encode("ascii")
    elif isinstance(counts, bytes):
        rle = segmentation
    else:
        raise BuildError(f"{context} RLE counts must be a list, string, or bytes")
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2).astype(np.uint8)
    return mask.astype(np.uint8, copy=False)


def _encode_mask(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": list(encoded["size"]), "counts": counts}


def _mask_bbox_and_area(mask: np.ndarray) -> tuple[list[float], float]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    bbox = [float(value) for value in mask_utils.toBbox(encoded).tolist()]
    area = float(mask_utils.area(encoded))
    return bbox, area


def _scale_polygon_segmentation(segmentation: list[Any], scale: float, context: str) -> list[list[float]]:
    scaled_polygons: list[list[float]] = []
    for polygon_idx, polygon in enumerate(segmentation):
        if not isinstance(polygon, list):
            raise BuildError(f"{context} polygon {polygon_idx} must be a list")
        if len(polygon) < 6 or len(polygon) % 2 != 0:
            raise BuildError(f"{context} polygon {polygon_idx} must contain an even number of at least 6 coordinates")
        scaled_polygons.append([_require_number(value, f"{context} polygon coordinate") * scale for value in polygon])
    return scaled_polygons


def _scale_bbox(raw_bbox: Any, scale: float, context: str) -> list[float]:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise BuildError(f"{context} bbox must be COCO [x, y, width, height]")
    return [_require_number(value, f"{context} bbox[{idx}]") * scale for idx, value in enumerate(raw_bbox)]


def _scale_annotation(
    ann: dict[str, Any],
    *,
    source_height: int,
    source_width: int,
    output_height: int,
    output_width: int,
    scale: float,
) -> dict[str, Any]:
    ann_id = _require_int(ann.get("id"), "annotation.id")
    context = f"annotation {ann_id}"
    if "segmentation" not in ann:
        raise BuildError(f"{context} missing segmentation")
    if "bbox" not in ann:
        raise BuildError(f"{context} missing bbox")
    if "area" not in ann:
        raise BuildError(f"{context} missing area")

    out = dict(ann)
    segmentation = ann["segmentation"]
    if isinstance(segmentation, list):
        out["segmentation"] = _scale_polygon_segmentation(segmentation, scale, context)
        out["bbox"] = _scale_bbox(ann["bbox"], scale, context)
        out["area"] = _require_number(ann["area"], f"{context} area") * scale * scale
    elif isinstance(segmentation, dict):
        mask = _decode_rle(segmentation, source_height, source_width, context)
        resized = cv2.resize(mask, (output_width, output_height), interpolation=cv2.INTER_NEAREST)
        resized = (resized > 0).astype(np.uint8, copy=False)
        out["segmentation"] = _encode_mask(resized)
        out["bbox"], out["area"] = _mask_bbox_and_area(resized)
    else:
        raise BuildError(f"{context} segmentation must be COCO polygon or RLE")
    return out


def _quantile(values: list[float], percent: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return float(finite[0])
    rank = (len(finite) - 1) * percent / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(finite[low])
    return float(finite[low] + (finite[high] - finite[low]) * (rank - low))


def build_resized_coco_rgbd_dataset(
    *,
    source_root: Path,
    source_ann: Path,
    output_root: Path,
    output_ann: Path,
    scale: float,
    max_images: int | None,
    split: str,
    summary_json: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if scale <= 0 or not math.isfinite(scale):
        raise BuildError("--scale must be a positive finite number")
    if max_images is not None and max_images <= 0:
        raise BuildError("--max-images must be positive when provided")

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    output_ann = output_ann.resolve()
    summary_json = summary_json.resolve()
    _prepare_outputs(output_root, output_ann, summary_json, overwrite=overwrite)

    coco = _load_json(source_ann)
    images = _require_list(coco, "images")
    annotations = _require_list(coco, "annotations")
    categories = _require_list(coco, "categories")

    selected_images = images[:max_images] if max_images is not None else list(images)
    image_by_id: dict[int, dict[str, Any]] = {}
    selected_ids: set[int] = set()
    for raw_image in selected_images:
        if not isinstance(raw_image, dict):
            raise BuildError("COCO images entries must be objects")
        image_id = _require_int(raw_image.get("id"), "image.id")
        if image_id in image_by_id:
            raise BuildError(f"Duplicate image id in selected images: {image_id}")
        _require_int(raw_image.get("width"), f"image {image_id}.width")
        _require_int(raw_image.get("height"), f"image {image_id}.height")
        file_name = raw_image.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise BuildError(f"image {image_id}.file_name must be a non-empty string")
        image_by_id[image_id] = raw_image
        selected_ids.add(image_id)

    selected_annotations: list[dict[str, Any]] = []
    for raw_ann in annotations:
        if not isinstance(raw_ann, dict):
            raise BuildError("COCO annotations entries must be objects")
        image_id = _require_int(raw_ann.get("image_id"), "annotation.image_id")
        if image_id in selected_ids:
            selected_annotations.append(raw_ann)

    output_images: list[dict[str, Any]] = []
    output_annotations: list[dict[str, Any]] = []
    file_counts = {"rgb": 0, "depth_npy": 0}
    written_rgb: set[Path] = set()
    written_depth: set[Path] = set()

    anns_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in selected_ids}
    for ann in selected_annotations:
        anns_by_image[int(ann["image_id"])].append(ann)

    for image in selected_images:
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        source_width = int(image["width"])
        source_height = int(image["height"])
        output_width, output_height = _scaled_size(source_width, source_height, scale)

        src_rgb = source_root / "images" / split / file_name
        dst_rgb = output_root / "images" / split / file_name
        stem = Path(file_name).stem
        src_depth = source_root / "depth" / "depth_npy" / split / f"{stem}.npy"
        dst_depth = output_root / "depth" / "depth_npy" / split / f"{stem}.npy"
        if dst_rgb in written_rgb or dst_depth in written_depth:
            raise BuildError(f"Output filename collision for image file_name={file_name!r}")

        _resize_rgb(src_rgb, dst_rgb, output_width, output_height)
        _resize_depth(src_depth, dst_depth, output_width, output_height)
        written_rgb.add(dst_rgb)
        written_depth.add(dst_depth)
        file_counts["rgb"] += 1
        file_counts["depth_npy"] += 1

        out_image = dict(image)
        out_image["width"] = output_width
        out_image["height"] = output_height
        output_images.append(out_image)

        for ann in anns_by_image.get(image_id, []):
            output_annotations.append(
                _scale_annotation(
                    ann,
                    source_height=source_height,
                    source_width=source_width,
                    output_height=output_height,
                    output_width=output_width,
                    scale=scale,
                )
            )

    output_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": output_images,
        "annotations": output_annotations,
        "categories": categories,
    }
    _write_json(output_ann, output_coco)

    areas = [float(ann["area"]) for ann in output_annotations]
    summary = {
        "source_root": str(source_root),
        "source_ann": str(source_ann),
        "output_root": str(output_root),
        "output_ann": str(output_ann),
        "split": split,
        "scale": scale,
        "max_images": max_images,
        "images": len(output_images),
        "annotations": len(output_annotations),
        "mask_area_p50": _quantile(areas, 50),
        "area_le_256_ratio": (sum(1 for area in areas if area <= 256.0) / len(areas)) if areas else None,
        "file_existence_counts": file_counts,
        "depth_resize_interpolation": "cv2.INTER_LINEAR; float depth values are preserved without normalization",
        "rgb_resize_interpolation": "PIL.Image.Resampling.BILINEAR",
        "polygon_policy": "scale coordinates, bbox, and area numerically by scale",
        "rle_policy": "decode mask, resize with cv2.INTER_NEAREST, encode compressed COCO RLE, recompute bbox and area from resized mask",
    }
    _write_json(summary_json, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-ann", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-ann", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--split", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--overwrite", action="store_true", help="replace existing output-root/output-ann/summary-json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    source_ann = _resolve_input_ann(source_root, args.source_ann)
    output_ann = _resolve_output_ann(output_root, args.output_ann)
    summary_json = Path(args.summary_json).expanduser()
    if not summary_json.is_absolute():
        summary_json = (Path.cwd() / summary_json).resolve()

    try:
        summary = build_resized_coco_rgbd_dataset(
            source_root=source_root,
            source_ann=source_ann,
            output_root=output_root,
            output_ann=output_ann,
            scale=float(args.scale),
            max_images=args.max_images,
            split=str(args.split),
            summary_json=summary_json,
            overwrite=bool(args.overwrite),
        )
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
