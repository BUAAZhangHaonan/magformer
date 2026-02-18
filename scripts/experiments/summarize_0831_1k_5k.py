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
        "best": pack(best),
        "last": pack(last),
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
        "best": pack(best),
        "last": pack(last),
    }


def _summarize_yolo(out_dir: Path) -> Dict[str, Any]:
    # Ultralytics writes runs under OUT/train by default with our runner.
    p = out_dir / "train" / "results.csv"
    if not p.exists():
        return {"status": "missing", "path": str(p)}

    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {"status": "empty", "path": str(p)}

    last = rows[-1]

    # Column names vary across ultralytics versions; handle common ones.
    def get_any(keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in last and last[k] != "":
                try:
                    return float(last[k])
                except Exception:
                    continue
        return None

    box_map5095 = get_any(["metrics/mAP50-95(B)", "metrics/mAP50-95"])
    box_map50 = get_any(["metrics/mAP50(B)"])
    mask_map5095 = get_any(["metrics/mAP50-95(M)"])
    mask_map50 = get_any(["metrics/mAP50(M)"])

    return {
        "status": "ok",
        "wall_time_sec": _read_wall_time(out_dir),
        "last": {
            "epoch": int(float(last.get("epoch", -1))),
            "bbox": {"AP": pct(box_map5095), "AP50": pct(box_map50)},
            "segm": {"AP": pct(mask_map5095), "AP50": pct(mask_map50)},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=str, required=True)
    ap.add_argument("--write", action="store_true", help="Write summary json under output root.")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    summary = {
        "experiment": "0831_1k_5k",
        "output_root": str(out_root),
        "magformer": _summarize_magformer(out_root / "magformer"),
        "mgm_mask2former": _summarize_detectron2(out_root / "mgm_mask2former"),
        "official_mask2former": _summarize_detectron2(out_root / "official_mask2former"),
        "maskrcnn": _summarize_detectron2(out_root / "maskrcnn"),
        "yolov8_seg": _summarize_yolo(out_root / "yolov8_seg"),
    }

    if args.write:
        p = out_root / "summary_0831_1k_5k.json"
        out_root.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[summary] wrote: {p}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

