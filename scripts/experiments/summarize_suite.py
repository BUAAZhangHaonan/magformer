#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ffloat(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v


def _pct01_to_100(x: Any) -> Optional[float]:
    v = _ffloat(x)
    if v is None:
        return None
    return v * 100.0


def _pack_from_cocoeval(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "iter": int(row.get("iteration", -1)),
        "segm": {
            "AP": _ffloat(row.get("segm/AP")),
            "AP50": _ffloat(row.get("segm/AP50")),
            "AP75": _ffloat(row.get("segm/AP75")),
            "APs": _ffloat(row.get("segm/APs")),
            "APm": _ffloat(row.get("segm/APm")),
            "APl": _ffloat(row.get("segm/APl")),
        },
        "bbox": {
            "AP": _ffloat(row.get("bbox/AP")),
            "AP50": _ffloat(row.get("bbox/AP50")),
            "AP75": _ffloat(row.get("bbox/AP75")),
            "APs": _ffloat(row.get("bbox/APs")),
            "APm": _ffloat(row.get("bbox/APm")),
            "APl": _ffloat(row.get("bbox/APl")),
        },
    }


def _pack_from_detectron2_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "iter": int(row.get("iteration", -1)),
        "segm": {
            "AP": _ffloat(row.get("segm/AP")),
            "AP50": _ffloat(row.get("segm/AP50")),
            "AP75": _ffloat(row.get("segm/AP75")),
            "APs": _ffloat(row.get("segm/APs")),
            "APm": _ffloat(row.get("segm/APm")),
            "APl": _ffloat(row.get("segm/APl")),
        },
        "bbox": {
            "AP": _ffloat(row.get("bbox/AP")),
            "AP50": _ffloat(row.get("bbox/AP50")),
            "AP75": _ffloat(row.get("bbox/AP75")),
            "APs": _ffloat(row.get("bbox/APs")),
            "APm": _ffloat(row.get("bbox/APm")),
            "APl": _ffloat(row.get("bbox/APl")),
        },
    }


def _pack_from_magformer_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "iter": int(row.get("iter", -1)),
        "segm": {
            "AP": _pct01_to_100(row.get("val/segm_AP")),
            "AP50": _pct01_to_100(row.get("val/segm_AP50")),
            "AP75": _pct01_to_100(row.get("val/segm_AP75")),
            "APs": _pct01_to_100(row.get("val/segm_APs", row.get("val/segm_AP_small"))),
            "APm": _pct01_to_100(row.get("val/segm_APm", row.get("val/segm_AP_medium"))),
            "APl": _pct01_to_100(row.get("val/segm_APl", row.get("val/segm_AP_large"))),
        },
        "bbox": {
            "AP": _pct01_to_100(row.get("val/bbox_AP")),
            "AP50": _pct01_to_100(row.get("val/bbox_AP50")),
            "AP75": _pct01_to_100(row.get("val/bbox_AP75")),
            "APs": _pct01_to_100(row.get("val/bbox_APs", row.get("val/bbox_AP_small"))),
            "APm": _pct01_to_100(row.get("val/bbox_APm", row.get("val/bbox_AP_medium"))),
            "APl": _pct01_to_100(row.get("val/bbox_APl", row.get("val/bbox_AP_large"))),
        },
    }


def _best_from_detectron2(out_dir: Path) -> Optional[Dict[str, Any]]:
    rows = _load_jsonl(out_dir / "metrics.json")
    val = [r for r in rows if "segm/AP" in r and "iteration" in r]
    if not val:
        return None
    best = max(val, key=lambda r: float(r.get("segm/AP", -1e9)))
    return _pack_from_detectron2_metrics(best)


def _best_from_magformer(out_dir: Path) -> Optional[Dict[str, Any]]:
    rows = _load_jsonl(out_dir / "metrics_log.jsonl")
    val = [r for r in rows if r.get("phase") == "val" and "val/segm_AP" in r]
    if not val:
        return None
    best = max(val, key=lambda r: float(r.get("val/segm_AP", -1e9)))
    return _pack_from_magformer_metrics(best)


def _best_from_yolo_csv(out_dir: Path) -> Optional[Dict[str, Any]]:
    csv_path = out_dir / "train" / "results.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def _get_any(row: Dict[str, str], keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in row and row[k] != "":
                try:
                    return float(row[k])
                except Exception:
                    continue
        return None

    def _score(row: Dict[str, str]) -> float:
        v = _get_any(row, ["metrics/mAP50-95(M)"])
        return float(v) if v is not None else -1e9

    best_row = max(rows, key=_score)
    return {
        "epoch": int(float(best_row.get("epoch", -1))),
        "segm": {
            "AP": _pct01_to_100(_get_any(best_row, ["metrics/mAP50-95(M)"])),
            "AP50": _pct01_to_100(_get_any(best_row, ["metrics/mAP50(M)"])),
            "AP75": None,
            "APs": None,
            "APm": None,
            "APl": None,
        },
        "bbox": {
            "AP": _pct01_to_100(_get_any(best_row, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])),
            "AP50": _pct01_to_100(_get_any(best_row, ["metrics/mAP50(B)"])),
            "AP75": None,
            "APs": None,
            "APm": None,
            "APl": None,
        },
    }


def _read_metadata(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "metadata.json"
    if not p.exists():
        return {}
    try:
        payload = _load_json(p)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_wall_time(out_dir: Path) -> Optional[float]:
    p = out_dir / "wall_time_sec.txt"
    if not p.exists():
        meta = _read_metadata(out_dir)
        wall_time = _ffloat(meta.get("wall_time_sec"))
        if wall_time is not None:
            return wall_time
        start = meta.get("start_time_iso")
        end = meta.get("end_time_iso")
        if isinstance(start, str) and isinstance(end, str):
            try:
                return float((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
            except Exception:
                return None
        return None
    try:
        return float(p.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _read_params(out_dir: Path) -> Optional[int]:
    p = out_dir / "params_trainable.txt"
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _read_peak_memory(out_dir: Path) -> Optional[float]:
    p = out_dir / "peak_memory_mb.txt"
    if not p.exists():
        best: Optional[float] = None
        for cand in [out_dir / "log.txt", out_dir / "run.log"]:
            if not cand.exists():
                continue
            text = cand.read_text(encoding="utf-8", errors="ignore")
            for match in re.finditer(r"max_mem:\s*(\d+)M", text):
                best = max(best or 0.0, float(match.group(1)))
            for match in re.finditer(r"\b\d+/\d+\s+([0-9]+(?:\.[0-9]+)?)G\b", text):
                best = max(best or 0.0, float(match.group(1)) * 1024.0)
        if best is not None:
            return best
        inference = _read_inference_speed(out_dir)
        if isinstance(inference, dict):
            inferred_peak = _ffloat(inference.get("inference_peak_memory_mb"))
            if inferred_peak is not None:
                return inferred_peak
        return None
    try:
        return float(p.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _read_inference_speed(out_dir: Path) -> Optional[Dict[str, Any]]:
    for name in ["inference_speed_clean.json", "inference_speed.json"]:
        p = out_dir / name
        if not p.exists():
            continue
        try:
            payload = _load_json(p)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _guess_framework(out_dir: Path) -> str:
    if (out_dir / "metrics_log.jsonl").exists():
        return "magformer"
    if (out_dir / "metrics.json").exists():
        return "detectron2"
    if (out_dir / "train" / "results.csv").exists():
        return "yolo"
    return "unknown"


def _summarize_model(out_dir: Path, framework: str) -> Dict[str, Any]:
    coco_metrics_path = out_dir / "metrics.cocoeval.json"
    coco_results_path = out_dir / "coco_instances_results.json"

    if not coco_metrics_path.exists():
        # Legacy fallback for unfinished runs.
        if framework == "yolo":
            fallback_best = _best_from_yolo_csv(out_dir)
            if fallback_best is not None:
                return {
                    "status": "partial",
                    "metrics_source": "ultralytics",
                    "wall_time_sec": _read_wall_time(out_dir),
                    "peak_memory_mb": _read_peak_memory(out_dir),
                    "inference": _read_inference_speed(out_dir),
                    "params_trainable": _read_params(out_dir),
                    "best": fallback_best,
                    "last": fallback_best,
                    "artifacts": {
                        "metrics": str(out_dir / "train" / "results.csv"),
                        "coco_instances_results": str(coco_results_path),
                    },
                }
        return {"status": "missing", "path": str(coco_metrics_path)}

    coco_row = _load_json(coco_metrics_path)
    last = _pack_from_cocoeval(coco_row)
    publication = last
    training_progress_best = None

    if framework == "magformer":
        training_progress_best = _best_from_magformer(out_dir)
        best = training_progress_best or publication
    elif framework == "detectron2":
        # For publication we trust the final exported COCOeval artifact, not the
        # online trainer stream, because some detectron2-style families do not
        # persist complete bbox metrics in metrics.json.
        training_progress_best = _best_from_detectron2(out_dir)
        best = publication
    elif framework == "yolo":
        # Keep canonical COCOeval as source of truth for `last`.
        training_progress_best = _best_from_yolo_csv(out_dir)
        best = training_progress_best or publication
    else:
        best = publication

    summary = {
        "status": "ok",
        "metrics_source": "cocoeval",
        "publication_metrics_source": "cocoeval_final",
        "wall_time_sec": _read_wall_time(out_dir),
        "peak_memory_mb": _read_peak_memory(out_dir),
        "inference": _read_inference_speed(out_dir),
        "params_trainable": _read_params(out_dir),
        "publication": publication,
        "best": best,
        "last": last,
        "artifacts": {
            "metrics": str(coco_metrics_path),
            "coco_instances_results": str(coco_results_path),
        },
    }
    if training_progress_best is not None:
        summary["training_progress_best"] = training_progress_best
    return summary


def _auto_scan_models(output_root: Path) -> List[str]:
    ignore = {"_tuning", "visualizations"}
    out: List[str] = []
    if not output_root.exists():
        return out
    for p in sorted(output_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name in ignore:
            continue
        if p.name.startswith("."):
            continue
        if p.name.startswith("_"):
            continue
        out.append(p.name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=str, required=True)
    ap.add_argument("--models-manifest", type=str, default=None, help="Optional JSON with {'models':[{'id','framework'}]}")
    ap.add_argument("--write", action="store_true", help="Write summary_<experiment>.json under output root.")
    ap.add_argument(
        "--write-name",
        type=str,
        default="",
        help="Optional custom summary file name when --write is set.",
    )
    args = ap.parse_args()

    out_root = Path(args.output_root).resolve()
    experiment = out_root.name

    models: List[Dict[str, str]] = []
    if args.models_manifest:
        manifest = _load_json(Path(args.models_manifest).resolve())
        for item in manifest.get("models", []):
            mid = str(item.get("id", "")).strip()
            if not mid:
                continue
            models.append({"id": mid, "framework": str(item.get("framework") or "").strip()})
    else:
        for mid in _auto_scan_models(out_root):
            models.append({"id": mid, "framework": ""})

    summary: Dict[str, Any] = {"experiment": experiment, "output_root": str(out_root)}
    for item in models:
        mid = item["id"]
        out_dir = out_root / mid
        framework = item["framework"] or _guess_framework(out_dir)
        summary[mid] = _summarize_model(out_dir, framework=framework)

    if args.write:
        out_root.mkdir(parents=True, exist_ok=True)
        p = out_root / (args.write_name or f"summary_{experiment}.json")
        p.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[summary] wrote: {p}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
