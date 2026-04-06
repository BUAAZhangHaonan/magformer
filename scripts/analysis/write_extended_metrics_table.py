#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


DEFAULT_SUMMARIES = [
    "output/experiments/0831_1k_20ep_1024_depth_revisit/summary_0831_1k_20ep_1024_depth_revisit.json",
    "output/experiments/0831_1k_20ep_1024_lightdepth_stage_a/summary_0831_1k_20ep_1024_lightdepth_stage_a.json",
    "output/experiments/0831_1k_20ep_1024_lightdepth_stage_b/summary_0831_1k_20ep_1024_lightdepth_stage_b.json",
    "output/experiments/0831_1k_20ep_1024_lightdepth_stage_b_f5/summary_0831_1k_20ep_1024_lightdepth_stage_b_f5.json",
]

MANIFEST_HEADERS = [
    "resolution",
    "model_id",
    "training_mode",
    "status",
    "segm_AP",
    "segm_AP50",
    "segm_AP75",
    "bbox_AP",
    "bbox_AP50",
    "bbox_AP75",
    "segm_precision_at_50",
    "segm_recall_at_50",
    "segm_f1_at_50",
    "params_trainable",
    "train_wall_time_sec",
    "peak_memory_mb",
    "inference_latency_ms_mean",
    "inference_peak_memory_mb",
    "inference_fps",
    "note",
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(block: Dict[str, Any] | None, family: str, key: str) -> Any:
    if not isinstance(block, dict):
        return None
    metric_block = block.get(family)
    if not isinstance(metric_block, dict):
        return None
    return metric_block.get(key)


def _coalesce_entry(current: Dict[str, Any] | None, candidate: Dict[str, Any]) -> Dict[str, Any]:
    if current is None:
        return candidate
    score_current = sum(v is not None for v in current.values())
    score_candidate = sum(v is not None for v in candidate.values())
    if score_candidate > score_current:
        return candidate
    return current


def _metric_top_level(entry: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = entry.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _canonical_model_id(model_id: str, payload: Dict[str, Any]) -> str:
    if model_id == "msmformer_scratch" and "msmformer" in payload:
        return "msmformer"
    if not (model_id.startswith("yolov8_") and model_id.endswith("_pretrained")):
        return model_id
    base_model_id = model_id[: -len("_pretrained")]
    if base_model_id in payload:
        return base_model_id
    return model_id


def _model_dir(summary_path: Path, model_id: str) -> Path:
    return summary_path.resolve().parent / model_id


def _read_dataset_root(summary_path: Path, model_id: str) -> Optional[Path]:
    metadata_path = _model_dir(summary_path, model_id) / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        payload = _load_json(metadata_path)
    except Exception:
        return None
    dataset_root = payload.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root.strip():
        return None
    return Path(dataset_root).resolve()


@lru_cache(maxsize=None)
def _compute_prf50_from_paths(annotation_path_str: str, results_path_str: str) -> Dict[str, Optional[float]]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception:
        return {"precision": None, "recall": None, "f1": None}

    annotation_path = Path(annotation_path_str)
    results_path = Path(results_path_str)
    if not annotation_path.exists() or not results_path.exists():
        return {"precision": None, "recall": None, "f1": None}

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(annotation_path))
    except Exception:
        return {"precision": None, "recall": None, "f1": None}

    try:
        rows = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return {"precision": None, "recall": None, "f1": None}
    if not isinstance(rows, list):
        return {"precision": None, "recall": None, "f1": None}

    total_gt = sum(1 for ann in coco_gt.dataset.get("annotations", []) if int(ann.get("iscrowd", 0)) == 0)
    if total_gt == 0:
        return {"precision": None, "recall": None, "f1": None}
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(str(results_path))
            evaluator = COCOeval(coco_gt, coco_dt, "segm")
            evaluator.params.iouThrs = np.array([0.5], dtype=np.float64)
            evaluator.params.maxDets = [100]
            evaluator.evaluate()
            evaluator.accumulate()
    except Exception:
        return {"precision": None, "recall": None, "f1": None}

    precision = evaluator.eval.get("precision")
    if precision is None:
        return {"precision": None, "recall": None, "f1": None}
    # Shape: [T, R, K, A, M]. We use IoU=0.5, area=all, maxDets=100 and average over categories.
    precision_slice = precision[0, :, :, 0, 0]
    if precision_slice.ndim == 1:
        precision_slice = precision_slice[:, None]
    valid = precision_slice > -1
    if not np.any(valid):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    mean_precision = np.zeros(precision_slice.shape[0], dtype=np.float64)
    for idx in range(precision_slice.shape[0]):
        vals = precision_slice[idx][valid[idx]]
        mean_precision[idx] = float(vals.mean()) if vals.size else 0.0

    recall_thresholds = evaluator.params.recThrs
    denom = mean_precision + recall_thresholds
    f1_curve = np.divide(
        2.0 * mean_precision * recall_thresholds,
        denom,
        out=np.zeros_like(mean_precision, dtype=np.float64),
        where=denom > 0,
    )
    best_idx = int(np.argmax(f1_curve))
    return {
        "precision": float(mean_precision[best_idx] * 100.0),
        "recall": float(recall_thresholds[best_idx] * 100.0),
        "f1": float(f1_curve[best_idx] * 100.0),
    }


def _segm_prf50(summary_path: Path, model_id: str, entry: Dict[str, Any]) -> Dict[str, Optional[float]]:
    direct_precision = _metric_top_level(entry, "segm_precision_at_50", "precision_at_50")
    direct_recall = _metric_top_level(entry, "segm_recall_at_50", "recall_at_50")
    direct_f1 = _metric_top_level(entry, "segm_f1_at_50", "f1_at_50")
    if direct_precision is not None or direct_recall is not None or direct_f1 is not None:
        return {"precision": direct_precision, "recall": direct_recall, "f1": direct_f1}

    dataset_root = _read_dataset_root(summary_path, model_id)
    if dataset_root is None:
        return {"precision": None, "recall": None, "f1": None}
    annotation_path = dataset_root / "annotations" / "instances_val.json"
    results_path = _model_dir(summary_path, model_id) / "coco_instances_results.json"
    return _compute_prf50_from_paths(str(annotation_path), str(results_path))


def _build_rows(summary_paths: List[Path]) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for path in summary_paths:
        payload = _load_json(path)
        for model_id, entry in payload.items():
            if model_id in {"experiment", "output_root"} or not isinstance(entry, dict):
                continue
            if entry.get("status") not in {"ok", "partial"}:
                continue
            canonical_model_id = _canonical_model_id(model_id, payload)
            best = entry.get("best") or {}
            last = entry.get("last") or {}
            inference = entry.get("inference") or {}
            prf50 = _segm_prf50(path, model_id, entry)
            candidate = {
                "model_id": canonical_model_id,
                "status": entry.get("status"),
                "best_segm_AP": _metric(best, "segm", "AP"),
                "best_segm_AP50": _metric(best, "segm", "AP50"),
                "best_segm_AP75": _metric(best, "segm", "AP75"),
                "best_segm_APs": _metric(best, "segm", "APs"),
                "best_segm_APm": _metric(best, "segm", "APm"),
                "best_bbox_AP": _metric(best, "bbox", "AP"),
                "best_bbox_AP50": _metric(best, "bbox", "AP50"),
                "best_bbox_AP75": _metric(best, "bbox", "AP75"),
                "best_bbox_APs": _metric(best, "bbox", "APs"),
                "best_bbox_APm": _metric(best, "bbox", "APm"),
                "segm_precision_at_50": prf50.get("precision"),
                "segm_recall_at_50": prf50.get("recall"),
                "segm_f1_at_50": prf50.get("f1"),
                "last_segm_AP": _metric(last, "segm", "AP"),
                "last_bbox_AP": _metric(last, "bbox", "AP"),
                "params_trainable": entry.get("params_trainable"),
                "train_wall_time_sec": entry.get("wall_time_sec"),
                "peak_memory_mb": entry.get("peak_memory_mb"),
                "inference_latency_ms_mean": inference.get("latency_ms_mean"),
                "inference_latency_ms_p50": inference.get("latency_ms_p50"),
                "inference_latency_ms_p90": inference.get("latency_ms_p90"),
                "inference_fps": inference.get("throughput_fps"),
                "inference_peak_memory_mb": inference.get("inference_peak_memory_mb"),
                "inference_status": inference.get("status"),
            }
            rows[canonical_model_id] = _coalesce_entry(rows.get(canonical_model_id), candidate)
    return sorted(
        rows.values(),
        key=lambda row: (
            row["best_segm_AP"] is None,
            -(row["best_segm_AP"] or -1.0),
            -(row["last_segm_AP"] or -1.0),
            -(row["inference_fps"] or -1.0),
            row["model_id"],
        ),
    )


def _to_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Model",
        "Best segm AP",
        "segm AP50",
        "segm AP75",
        "APs",
        "APm",
        "Best bbox AP",
        "Best bbox AP50",
        "Best bbox AP75",
        "P@50",
        "R@50",
        "F1@50",
        "Last segm AP",
        "Params",
        "Train sec",
        "Peak mem MB",
        "Infer ms",
        "Infer FPS",
        "Infer status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["model_id"],
            row["best_segm_AP"],
            row["best_segm_AP50"],
            row["best_segm_AP75"],
            row["best_segm_APs"],
            row["best_segm_APm"],
            row["best_bbox_AP"],
            row["best_bbox_AP50"],
            row["best_bbox_AP75"],
            row["segm_precision_at_50"],
            row["segm_recall_at_50"],
            row["segm_f1_at_50"],
            row["last_segm_AP"],
            row["params_trainable"],
            row["train_wall_time_sec"],
            row["peak_memory_mb"],
            row["inference_latency_ms_mean"],
            row["inference_fps"],
            row["inference_status"],
        ]
        formatted = []
        for value in values:
            if isinstance(value, float):
                formatted.append(f"{value:.4f}")
            elif value is None:
                formatted.append("")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines) + "\n"


def _load_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("rows")
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a JSON list or an object with rows: {path}")
    rows: List[Dict[str, Any]] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        row = {key: raw.get(key) for key in MANIFEST_HEADERS}
        row["row_index"] = raw.get("row_index")
        rows.append(row)
    return rows


def _manifest_sort_key(row: Dict[str, Any]) -> tuple[Any, ...]:
    resolution_order = {1024: 0, 512: 1, 256: 2}
    row_index = row.get("row_index")
    if row_index is not None:
        try:
            return (-1, int(row_index))
        except Exception:
            pass
    return (
        resolution_order.get(int(row.get("resolution") or 0), 999),
        str(row.get("model_id") or ""),
    )


def _to_manifest_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Resolution",
        "Model",
        "Training mode",
        "Status",
        "segm AP",
        "AP50",
        "AP75",
        "bbox AP",
        "bbox AP50",
        "bbox AP75",
        "P@50",
        "R@50",
        "F1@50",
        "Params",
        "Train sec",
        "Train mem MB",
        "Infer ms",
        "Infer mem MB",
        "FPS",
        "Note",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row.get("resolution"),
            row.get("model_id"),
            row.get("training_mode"),
            row.get("status"),
            row.get("segm_AP"),
            row.get("segm_AP50"),
            row.get("segm_AP75"),
            row.get("bbox_AP"),
            row.get("bbox_AP50"),
            row.get("bbox_AP75"),
            row.get("segm_precision_at_50"),
            row.get("segm_recall_at_50"),
            row.get("segm_f1_at_50"),
            row.get("params_trainable"),
            row.get("train_wall_time_sec"),
            row.get("peak_memory_mb"),
            row.get("inference_latency_ms_mean"),
            row.get("inference_peak_memory_mb"),
            row.get("inference_fps"),
            row.get("note"),
        ]
        formatted: List[str] = []
        for value in values:
            if isinstance(value, float):
                formatted.append(f"{value:.4f}")
            elif value is None:
                formatted.append("")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write extended metrics table from multiple summary jsons")
    parser.add_argument("--summary", action="append", default=[], help="Summary json path. Repeatable.")
    parser.add_argument("--manifest", default="", help="Live artifact manifest json.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest:
        manifest_rows = sorted(_load_manifest_rows(Path(args.manifest).resolve()), key=_manifest_sort_key)
        markdown = _to_manifest_markdown(manifest_rows)
        rows = [{key: row.get(key) for key in MANIFEST_HEADERS} for row in manifest_rows]
        fieldnames = MANIFEST_HEADERS
    else:
        summary_paths = [Path(path).resolve() for path in (args.summary or DEFAULT_SUMMARIES)]
        rows = _build_rows(summary_paths)
        markdown = _to_markdown(rows)
        fieldnames = list(rows[0].keys()) if rows else []

    out_json = Path(args.out_json).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_md = Path(args.out_md).resolve()
    for path in [out_json, out_csv, out_md]:
        path.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    out_md.write_text(markdown, encoding="utf-8")
    print(f"[extended-metrics] wrote: {out_json}")
    print(f"[extended-metrics] wrote: {out_csv}")
    print(f"[extended-metrics] wrote: {out_md}")


if __name__ == "__main__":
    main()
