#!/usr/bin/env python3
"""Visualize and summarize target_unlabeled prediction scale for R61."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

DEFAULT_TARGET_ANN = Path(
    "magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json"
)
DEFAULT_IMAGE_ROOT = Path("magformer_datasets/pseudo_real_512/images/train")
DEFAULT_OUTPUT_DIR = Path("output/diagnostics/r61_target_scale_visualization_20260516")
DEFAULT_EVAL_DIRS = {
    "Teacher": Path(
        "output/experiments/teacher_pseudoreal_target_unlabeled200_1024_backmap_recheck_20260515"
    ),
    "R59": Path("output/diagnostics/r59_retention_min_ckpt0250_target_unlabeled200_20260516"),
    "R46": Path(
        "output/diagnostics/r46_pseudoreal_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516"
    ),
}
MODEL_ORDER = ("Teacher", "R59", "R46")
CONTACT_COLUMNS = ("GT", "Teacher", "R59", "R46")
BOX_COLORS = {
    "GT": (60, 190, 90),
    "Teacher": (230, 80, 70),
    "R59": (240, 155, 45),
    "R46": (65, 180, 220),
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_prediction_file(eval_dir: str | Path) -> Path:
    """Return the COCO results JSON from an eval directory."""
    root = Path(eval_dir)
    preferred = root / "coco_instances_results.json"
    if preferred.is_file():
        return preferred
    candidates = sorted(path for path in root.glob("*.json") if "results" in path.name)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"no prediction JSON found under {root}")
    names = ", ".join(path.name for path in candidates)
    raise ValueError(f"ambiguous prediction JSON under {root}: {names}")


def _percentile(values: Sequence[float], q: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    pos = (len(finite) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return finite[lo]
    frac = pos - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def _bbox_area(bbox: Sequence[float]) -> float:
    if len(bbox) != 4:
        raise ValueError(f"bbox must have four values, got {bbox}")
    width = max(0.0, float(bbox[2]))
    height = max(0.0, float(bbox[3]))
    return width * height


def _bbox_iou_xywh(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = [float(value) for value in box_a]
    bx1, by1, bw, bh = [float(value) for value in box_b]
    ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
    bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = _bbox_area(box_a) + _bbox_area(box_b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _prediction_boxes(predictions: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    boxes: list[list[float]] = []
    for pred in predictions:
        bbox = pred.get("bbox")
        if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) == 4:
            boxes.append([float(value) for value in bbox])
    return boxes


def _best_iou_p50(
    gt_boxes: Sequence[Sequence[float]], pred_boxes: Sequence[Sequence[float]]
) -> float | None:
    if not gt_boxes or not pred_boxes:
        return None
    best_values = []
    for gt_box in gt_boxes:
        best_values.append(max(_bbox_iou_xywh(gt_box, pred_box) for pred_box in pred_boxes))
    return _percentile(best_values, 0.5)


def compute_image_stats(
    image_id: int,
    file_name: str,
    gt_boxes: Sequence[Sequence[float]],
    model_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Compute per-image GT scale and per-model prediction scale/IoU stats."""
    gt_areas = [_bbox_area(box) for box in gt_boxes]
    row: dict[str, Any] = {
        "image_id": int(image_id),
        "file_name": file_name,
        "gt_count": len(gt_boxes),
        "gt_bbox_area_p50": _percentile(gt_areas, 0.5),
        "models": {},
    }
    for model_name, predictions in model_predictions.items():
        pred_boxes = _prediction_boxes(predictions)
        pred_areas = [_bbox_area(box) for box in pred_boxes]
        scores = [
            float(pred.get("score", 0.0))
            for pred in predictions
            if isinstance(pred.get("score", 0.0), (int, float))
        ]
        row["models"][model_name] = {
            "pred_count": len(pred_boxes),
            "pred_bbox_area_p50": _percentile(pred_areas, 0.5),
            "pred_bbox_area_mean": float(mean(pred_areas)) if pred_areas else None,
            "best_iou_p50": _best_iou_p50(gt_boxes, pred_boxes),
            "score_p50": _percentile(scores, 0.5),
        }
    return row


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def select_images_for_contact_sheet(
    rows: Sequence[Mapping[str, Any]], limit: int = 10
) -> list[dict[str, Any]]:
    """Pick images where R46 matches GT while Teacher/R59 boxes are oversized."""

    def key(row: Mapping[str, Any]) -> tuple[bool, float, float, float, int]:
        gt_p50 = row.get("gt_bbox_area_p50")
        models = row.get("models", {})
        if not isinstance(models, Mapping):
            return (False, 0.0, 0.0, 0.0, 0)
        teacher = (
            models.get("Teacher", {}) if isinstance(models.get("Teacher", {}), Mapping) else {}
        )
        r59 = models.get("R59", {}) if isinstance(models.get("R59", {}), Mapping) else {}
        r46 = models.get("R46", {}) if isinstance(models.get("R46", {}), Mapping) else {}
        teacher_ratio = _safe_ratio(teacher.get("pred_bbox_area_p50"), gt_p50)  # type: ignore[arg-type]
        r59_ratio = _safe_ratio(r59.get("pred_bbox_area_p50"), gt_p50)  # type: ignore[arg-type]
        r46_iou = float(r46.get("best_iou_p50") or 0.0)
        min_oversize = min(teacher_ratio, r59_ratio)
        avg_oversize = (teacher_ratio + r59_ratio) / 2.0
        preferred = r46_iou >= 0.5 and min_oversize >= 2.0
        return (preferred, r46_iou, min_oversize, avg_oversize, -int(row.get("image_id", 0)))

    selected = sorted((dict(row) for row in rows), key=key, reverse=True)[:limit]
    return selected


def _load_coco(coco_path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[list[float]]]]:
    coco = _load_json(coco_path)
    images = {int(image["id"]): dict(image) for image in coco.get("images", [])}
    gt_by_image: dict[int, list[list[float]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        bbox = ann.get("bbox")
        if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) == 4:
            gt_by_image[int(ann["image_id"])].append([float(value) for value in bbox])
    return images, gt_by_image


def _load_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"prediction JSON must be a list: {path}")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"prediction row {idx} in {path} must be an object")
        if "image_id" not in item:
            raise ValueError(f"prediction row {idx} in {path} is missing image_id")
        grouped[int(item["image_id"])].append(dict(item))
    return grouped


def _build_all_stats(
    images: Mapping[int, Mapping[str, Any]],
    gt_by_image: Mapping[int, Sequence[Sequence[float]]],
    predictions_by_model: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    rows = []
    for image_id in sorted(images):
        file_name = str(images[image_id].get("file_name", ""))
        model_predictions = {
            model_name: predictions_by_model.get(model_name, {}).get(image_id, [])
            for model_name in MODEL_ORDER
        }
        rows.append(
            compute_image_stats(
                image_id, file_name, gt_by_image.get(image_id, []), model_predictions
            )
        )
    return rows


def _flatten_stats_for_csv(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flat_rows = []
    for row in rows:
        base = {
            "image_id": row["image_id"],
            "file_name": row["file_name"],
            "gt_count": row["gt_count"],
            "gt_bbox_area_p50": row["gt_bbox_area_p50"],
        }
        models = row.get("models", {})
        for model_name in MODEL_ORDER:
            stats = models.get(model_name, {}) if isinstance(models, Mapping) else {}
            for key in (
                "pred_count",
                "pred_bbox_area_p50",
                "pred_bbox_area_mean",
                "best_iou_p50",
                "score_p50",
            ):
                base[f"{model_name}_{key}"] = stats.get(key) if isinstance(stats, Mapping) else None
        flat_rows.append(base)
    return flat_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _median_of_field(
    rows: Sequence[Mapping[str, Any]], model_name: str, field: str
) -> float | None:
    values = []
    for row in rows:
        models = row.get("models", {})
        if isinstance(models, Mapping):
            stats = models.get(model_name, {})
            if isinstance(stats, Mapping) and stats.get(field) is not None:
                values.append(float(stats[field]))
    return _percentile(values, 0.5)


def _summary(
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    prediction_files: Mapping[str, Path],
) -> dict[str, Any]:
    gt_values = [
        float(row["gt_bbox_area_p50"]) for row in rows if row.get("gt_bbox_area_p50") is not None
    ]
    model_summary = {}
    for model_name in MODEL_ORDER:
        model_summary[model_name] = {
            "pred_bbox_area_p50_over_images": _median_of_field(
                rows, model_name, "pred_bbox_area_p50"
            ),
            "best_iou_p50_over_images": _median_of_field(rows, model_name, "best_iou_p50"),
            "pred_count_p50_over_images": _median_of_field(rows, model_name, "pred_count"),
        }
    return {
        "strategy": "Selected 10 images by sorting for R46 GT best-IoU p50 >= 0.5 and Teacher/R59 bbox area p50 at least 2x GT, then by R46 IoU and oversize ratio. If fewer qualify, the same deterministic sort fills the sheet.",
        "image_count": len(rows),
        "selected_image_ids": [int(row["image_id"]) for row in selected],
        "prediction_files": {name: str(path) for name, path in prediction_files.items()},
        "all_images": {
            "gt_bbox_area_p50_over_images": _percentile(gt_values, 0.5),
            "models": model_summary,
        },
    }


def _image_path(image_root: Path, file_name: str) -> Path:
    path = image_root / file_name
    if path.is_file():
        return path
    return image_root / Path(file_name).name


def _draw_boxes(
    draw: Any,
    boxes: Sequence[Sequence[float]],
    scale_x: float,
    scale_y: float,
    color: tuple[int, int, int],
    label: str,
    max_boxes: int,
) -> None:
    for idx, bbox in enumerate(boxes[:max_boxes]):
        x, y, w, h = [float(value) for value in bbox]
        xy = (x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y)
        draw.rectangle(xy, outline=color, width=2)
        if idx < 5:
            draw.text((xy[0] + 2, max(0, xy[1] - 10)), label, fill=color)


def _render_contact_sheet(
    *,
    path: Path,
    selected_rows: Sequence[Mapping[str, Any]],
    image_root: Path,
    images: Mapping[int, Mapping[str, Any]],
    gt_by_image: Mapping[int, Sequence[Sequence[float]]],
    predictions_by_model: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    panel_width: int,
    max_boxes: int,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    header_h = 34
    caption_h = 34
    gap = 6
    font = ImageFont.load_default()
    panels: list[list[Image.Image]] = []

    for row in selected_rows:
        image_id = int(row["image_id"])
        image_info = images[image_id]
        with Image.open(_image_path(image_root, str(image_info["file_name"]))) as src:
            base = src.convert("RGB")
        scale = panel_width / float(base.width)
        panel_height = max(1, int(round(base.height * scale)))
        resized = base.resize((panel_width, panel_height))
        row_panels = []
        for column in CONTACT_COLUMNS:
            panel = Image.new(
                "RGB", (panel_width, header_h + panel_height + caption_h), (245, 245, 245)
            )
            panel.paste(resized, (0, header_h))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel_width, header_h), fill=(30, 30, 30))
            draw.text((8, 9), f"{image_id} {column}", fill=(255, 255, 255), font=font)
            stats = row.get("models", {}).get(column, {}) if column != "GT" else {}
            if column == "GT":
                boxes = gt_by_image.get(image_id, [])
                label = "gt"
                gt_p50 = row.get("gt_bbox_area_p50")
                caption = f"n={len(boxes)} p50={gt_p50}"
            else:
                preds = predictions_by_model.get(column, {}).get(image_id, [])
                boxes = _prediction_boxes(preds)
                label = column.lower()
                pred_count = stats.get("pred_count")
                pred_p50 = stats.get("pred_bbox_area_p50")
                best_iou = stats.get("best_iou_p50")
                caption = f"n={pred_count} p50={pred_p50} IoU={best_iou}"
            _draw_boxes(
                draw,
                boxes,
                scale,
                scale,
                BOX_COLORS[column],
                label,
                max_boxes=max_boxes,
            )
            draw.rectangle(
                (0, header_h + panel_height, panel_width, header_h + panel_height + caption_h),
                fill=(255, 255, 255),
            )
            draw.text((8, header_h + panel_height + 8), caption[:54], fill=(20, 20, 20), font=font)
            row_panels.append(panel)
        panels.append(row_panels)

    if not panels:
        raise ValueError("no selected rows to render")
    row_width = len(CONTACT_COLUMNS) * panel_width + (len(CONTACT_COLUMNS) - 1) * gap
    row_height = panels[0][0].height
    sheet = Image.new(
        "RGB", (row_width, len(panels) * row_height + (len(panels) - 1) * gap), (220, 220, 220)
    )
    for row_idx, row_panels in enumerate(panels):
        y = row_idx * (row_height + gap)
        for col_idx, panel in enumerate(row_panels):
            x = col_idx * (panel_width + gap)
            sheet.paste(panel, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def run_diagnostic(
    *,
    target_ann: Path,
    image_root: Path,
    eval_dirs: Mapping[str, Path],
    output_dir: Path,
    limit: int,
    panel_width: int,
    max_boxes: int,
) -> dict[str, Any]:
    images, gt_by_image = _load_coco(target_ann)
    prediction_files = {
        name: discover_prediction_file(eval_dir) for name, eval_dir in eval_dirs.items()
    }
    predictions_by_model = {
        name: _load_predictions(path) for name, path in prediction_files.items()
    }
    all_rows = _build_all_stats(images, gt_by_image, predictions_by_model)
    selected = select_images_for_contact_sheet(all_rows, limit=limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_stats = {
        "summary": _summary(all_rows, selected, prediction_files),
        "images": selected,
    }
    (output_dir / "selected_stats.json").write_text(
        json.dumps(selected_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "all_image_stats.json").write_text(
        json.dumps(all_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "selected_stats.csv", _flatten_stats_for_csv(selected))
    _write_csv(output_dir / "all_image_stats.csv", _flatten_stats_for_csv(all_rows))
    (output_dir / "summary.json").write_text(
        json.dumps(selected_stats["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _render_contact_sheet(
        path=output_dir / "contact_sheet.png",
        selected_rows=selected,
        image_root=image_root,
        images=images,
        gt_by_image=gt_by_image,
        predictions_by_model=predictions_by_model,
        panel_width=panel_width,
        max_boxes=max_boxes,
    )
    return selected_stats["summary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-ann", type=Path, default=DEFAULT_TARGET_ANN)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--teacher-eval-dir", type=Path, default=DEFAULT_EVAL_DIRS["Teacher"])
    parser.add_argument("--r59-eval-dir", type=Path, default=DEFAULT_EVAL_DIRS["R59"])
    parser.add_argument("--r46-eval-dir", type=Path, default=DEFAULT_EVAL_DIRS["R46"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--panel-width", type=int, default=320)
    parser.add_argument("--max-boxes", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dirs = {
        "Teacher": args.teacher_eval_dir,
        "R59": args.r59_eval_dir,
        "R46": args.r46_eval_dir,
    }
    summary = run_diagnostic(
        target_ann=args.target_ann,
        image_root=args.image_root,
        eval_dirs=eval_dirs,
        output_dir=args.output_dir,
        limit=args.limit,
        panel_width=args.panel_width,
        max_boxes=args.max_boxes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
