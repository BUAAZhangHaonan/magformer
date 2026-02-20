#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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


def _read_wall_time(out_dir: Path) -> Optional[float]:
    p = out_dir / "wall_time_sec.txt"
    if not p.exists():
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

    if framework == "magformer":
        best = _best_from_magformer(out_dir) or last
    elif framework == "detectron2":
        best = _best_from_detectron2(out_dir) or last
    elif framework == "yolo":
        # Keep canonical COCOeval as source of truth for `last`.
        best = _best_from_yolo_csv(out_dir) or last
    else:
        best = last

    return {
        "status": "ok",
        "metrics_source": "cocoeval",
        "wall_time_sec": _read_wall_time(out_dir),
        "params_trainable": _read_params(out_dir),
        "best": best,
        "last": last,
        "artifacts": {
            "metrics": str(coco_metrics_path),
            "coco_instances_results": str(coco_results_path),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=str, required=True)
    ap.add_argument("--write", action="store_true", help="Write summary json under output root.")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    summary: Dict[str, Any] = {
        "experiment": "0831_1k_5k_scratch8",
        "output_root": str(out_root),
        "magformer_scratch": _summarize_model(out_root / "magformer_scratch", framework="magformer"),
        "mgm_mask2former_scratch": _summarize_model(out_root / "mgm_mask2former_scratch", framework="detectron2"),
        "msmformer_scratch": _summarize_model(out_root / "msmformer_scratch", framework="detectron2"),
        "uoais_scratch": _summarize_model(out_root / "uoais_scratch", framework="detectron2"),
        "ucn_scratch": _summarize_model(out_root / "ucn_scratch", framework="ucn"),
        "official_mask2former_scratch": _summarize_model(out_root / "official_mask2former_scratch", framework="detectron2"),
        "maskrcnn_scratch": _summarize_model(out_root / "maskrcnn_scratch", framework="detectron2"),
        "yolov8_seg_scratch": _summarize_model(out_root / "yolov8_seg_scratch", framework="yolo"),
    }

    if args.write:
        out_root.mkdir(parents=True, exist_ok=True)
        p = out_root / "summary_0831_1k_5k_scratch8.json"
        p.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[summary] wrote: {p}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
