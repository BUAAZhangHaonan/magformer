#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pct(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v * 100.0


def ffloat(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v


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


def _summarize_magformer(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "metrics_log.jsonl"
    if not p.exists():
        return {"status": "missing", "path": str(p)}
    rows = load_jsonl(p)
    val = [r for r in rows if r.get("phase") == "val" and "val/segm_AP" in r]
    if not val:
        return {"status": "no_val", "path": str(p)}

    best = max(val, key=lambda r: float(r.get("val/segm_AP", -1e9)))
    last = val[-1]

    def pack(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "iter": int(r.get("iter", -1)),
            "segm": {
                "AP": pct(r.get("val/segm_AP")),
                "AP50": pct(r.get("val/segm_AP50")),
                "AP75": pct(r.get("val/segm_AP75")),
                "APs": pct(r.get("val/segm_APs", r.get("val/segm_AP_small"))),
                "APm": pct(r.get("val/segm_APm", r.get("val/segm_AP_medium"))),
                "APl": pct(r.get("val/segm_APl", r.get("val/segm_AP_large"))),
            },
            "bbox": {
                "AP": pct(r.get("val/bbox_AP")),
                "AP50": pct(r.get("val/bbox_AP50")),
                "AP75": pct(r.get("val/bbox_AP75")),
                "APs": pct(r.get("val/bbox_APs", r.get("val/bbox_AP_small"))),
                "APm": pct(r.get("val/bbox_APm", r.get("val/bbox_AP_medium"))),
                "APl": pct(r.get("val/bbox_APl", r.get("val/bbox_AP_large"))),
            },
        }

    return {
        "status": "ok",
        "wall_time_sec": _read_wall_time(out_dir),
        "params_trainable": _read_params(out_dir),
        "best": pack(best),
        "last": pack(last),
        "artifacts": {
            "metrics": str(p),
            "coco_instances_results": str(out_dir / "coco_instances_results.json"),
        },
    }


def _summarize_detectron2(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "metrics.json"
    if not p.exists():
        return {"status": "missing", "path": str(p)}
    rows = load_jsonl(p)
    val = [r for r in rows if "segm/AP" in r or "bbox/AP" in r]
    if not val:
        return {"status": "no_val", "path": str(p)}

    def segm_ap(r: Dict[str, Any]) -> float:
        return float(r.get("segm/AP", -1e9))

    best = max(val, key=segm_ap)
    last = val[-1]

    def pack(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "iter": int(r.get("iteration", -1)),
            "segm": {
                "AP": ffloat(r.get("segm/AP")),
                "AP50": ffloat(r.get("segm/AP50")),
                "AP75": ffloat(r.get("segm/AP75")),
                "APs": ffloat(r.get("segm/APs")),
                "APm": ffloat(r.get("segm/APm")),
                "APl": ffloat(r.get("segm/APl")),
            },
            "bbox": {
                "AP": ffloat(r.get("bbox/AP")),
                "AP50": ffloat(r.get("bbox/AP50")),
                "AP75": ffloat(r.get("bbox/AP75")),
                "APs": ffloat(r.get("bbox/APs")),
                "APm": ffloat(r.get("bbox/APm")),
                "APl": ffloat(r.get("bbox/APl")),
            },
        }

    return {
        "status": "ok",
        "wall_time_sec": _read_wall_time(out_dir),
        "params_trainable": _read_params(out_dir),
        "best": pack(best),
        "last": pack(last),
        "artifacts": {
            "metrics": str(p),
            "coco_instances_results": str(out_dir / "inference" / "coco_instances_results.json"),
        },
    }


def _summarize_ucn(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "metrics.json"
    if not p.exists():
        return {"status": "missing", "path": str(p)}
    try:
        metrics = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "bad_json", "path": str(p)}

    pack = {
        "iter": int(metrics.get("iteration", -1)),
        "segm": {
            "AP": ffloat(metrics.get("segm/AP")),
            "AP50": ffloat(metrics.get("segm/AP50")),
            "AP75": ffloat(metrics.get("segm/AP75")),
            "APs": ffloat(metrics.get("segm/APs")),
            "APm": ffloat(metrics.get("segm/APm")),
            "APl": ffloat(metrics.get("segm/APl")),
        },
        "bbox": {
            "AP": ffloat(metrics.get("bbox/AP")),
            "AP50": ffloat(metrics.get("bbox/AP50")),
            "AP75": ffloat(metrics.get("bbox/AP75")),
            "APs": ffloat(metrics.get("bbox/APs")),
            "APm": ffloat(metrics.get("bbox/APm")),
            "APl": ffloat(metrics.get("bbox/APl")),
        },
    }

    return {
        "status": "ok",
        "wall_time_sec": _read_wall_time(out_dir),
        "params_trainable": _read_params(out_dir),
        "best": pack,
        "last": pack,
        "artifacts": {
            "metrics": str(p),
            "coco_instances_results": str(out_dir / "coco_instances_results.json"),
        },
    }


def _summarize_yolo(out_dir: Path) -> Dict[str, Any]:
    # Preferred: unified COCOeval results written by our runner.
    coco_metrics = out_dir / "metrics.json"
    coco_results = out_dir / "coco_instances_results.json"
    if coco_metrics.exists():
        rows = load_jsonl(coco_metrics)
        val = [r for r in rows if "segm/AP" in r or "bbox/AP" in r]
        if val:
            best = max(val, key=lambda r: float(r.get("segm/AP", -1e9)))
            last = val[-1]

            def pack(r: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "iter": int(r.get("iteration", -1)),
                    "segm": {
                        "AP": ffloat(r.get("segm/AP")),
                        "AP50": ffloat(r.get("segm/AP50")),
                        "AP75": ffloat(r.get("segm/AP75")),
                        "APs": ffloat(r.get("segm/APs")),
                        "APm": ffloat(r.get("segm/APm")),
                        "APl": ffloat(r.get("segm/APl")),
                    },
                    "bbox": {
                        "AP": ffloat(r.get("bbox/AP")),
                        "AP50": ffloat(r.get("bbox/AP50")),
                        "AP75": ffloat(r.get("bbox/AP75")),
                        "APs": ffloat(r.get("bbox/APs")),
                        "APm": ffloat(r.get("bbox/APm")),
                        "APl": ffloat(r.get("bbox/APl")),
                    },
                }

            return {
                "status": "ok",
                "metrics_source": "cocoeval",
                "wall_time_sec": _read_wall_time(out_dir),
                "params_trainable": _read_params(out_dir),
                "best": pack(best),
                "last": pack(last),
                "artifacts": {
                    "metrics": str(coco_metrics),
                    "coco_instances_results": str(coco_results),
                    "weights_best": str(out_dir / "train" / "weights" / "best.pt"),
                },
            }

    # Fallback: Ultralytics writes runs under OUT/train by default with our runner.
    p = out_dir / "train" / "results.csv"
    if not p.exists():
        return {"status": "missing", "path": str(p)}

    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {"status": "empty", "path": str(p)}

    # Column names vary across ultralytics versions; handle common ones.
    def get_any(row: Dict[str, str], keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in row and row[k] != "":
                try:
                    return float(row[k])
                except Exception:
                    continue
        return None

    def pack(row: Dict[str, str]) -> Dict[str, Any]:
        box_map5095 = get_any(row, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
        box_map50 = get_any(row, ["metrics/mAP50(B)"])
        mask_map5095 = get_any(row, ["metrics/mAP50-95(M)"])
        mask_map50 = get_any(row, ["metrics/mAP50(M)"])
        return {
            "epoch": int(float(row.get("epoch", -1))),
            "bbox": {"AP": pct(box_map5095), "AP50": pct(box_map50)},
            "segm": {"AP": pct(mask_map5095), "AP50": pct(mask_map50)},
        }

    def key_mask_ap(row: Dict[str, str]) -> float:
        v = get_any(row, ["metrics/mAP50-95(M)"])
        return float(v) if v is not None else -1e9

    best_row = max(rows, key=key_mask_ap)
    last_row = rows[-1]

    return {
        "status": "ok",
        "metrics_source": "ultralytics",
        "wall_time_sec": _read_wall_time(out_dir),
        "params_trainable": _read_params(out_dir),
        "best": pack(best_row),
        "last": pack(last_row),
        "artifacts": {
            "metrics": str(p),
            "weights_best": str(out_dir / "train" / "weights" / "best.pt"),
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
        "magformer_scratch": _summarize_magformer(out_root / "magformer_scratch"),
        "mgm_mask2former_scratch": _summarize_detectron2(out_root / "mgm_mask2former_scratch"),
        "msmformer_scratch": _summarize_detectron2(out_root / "msmformer_scratch"),
        "uoais_scratch": _summarize_detectron2(out_root / "uoais_scratch"),
        "ucn_scratch": _summarize_ucn(out_root / "ucn_scratch"),
        "official_mask2former_scratch": _summarize_detectron2(out_root / "official_mask2former_scratch"),
        "maskrcnn_scratch": _summarize_detectron2(out_root / "maskrcnn_scratch"),
        "yolov8_seg_scratch": _summarize_yolo(out_root / "yolov8_seg_scratch"),
    }

    if args.write:
        p = out_root / "summary_0831_1k_5k_scratch8.json"
        out_root.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[summary] wrote: {p}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
