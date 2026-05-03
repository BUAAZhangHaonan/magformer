#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FIELDS: List[str] = [
    "step",
    "step_unit",
    "step_iter",
    "step_epoch",
    "phase",
    "wall_time_iso",
    "elapsed_sec",
    "train/loss",
    "train/lr",
    "val/segm_AP",
    "val/segm_AP50",
    "val/segm_AP75",
    "val/segm_APs",
    "val/segm_APm",
    "val/segm_APl",
    "val/bbox_AP",
    "val/bbox_AP50",
    "val/bbox_AP75",
    "val/bbox_APs",
    "val/bbox_APm",
    "val/bbox_APl",
]


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


def _fint(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


def _pct01_to_100(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    # Most frameworks log COCO AP in [0, 100]. MAGFormer/YOLO tend to log [0, 1].
    if v <= 1.0 + 1e-6:
        return v * 100.0
    return v


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _detect_framework(out_dir: Path) -> str:
    if (out_dir / "metrics_log.jsonl").exists():
        return "magformer"
    metrics_json = out_dir / "metrics.json"
    if metrics_json.exists():
        # Detectron2 uses JSONL with "iteration" per line. Some baselines write a
        # single JSON dict to metrics.json; treat those as non-detectron2.
        try:
            for line in metrics_json.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and "iteration" in obj:
                    return "detectron2"
                break
        except Exception:
            pass
    if (out_dir / "train" / "results.csv").exists():
        return "yolo"
    if (out_dir / "metrics.cocoeval.json").exists():
        return "cocoeval"
    return "unknown"


def _blank() -> Dict[str, Any]:
    return {k: None for k in FIELDS}


def _finalize_step_fields(row: Dict[str, Any], iters_per_epoch: Optional[int]) -> None:
    step = _ffloat(row.get("step"))
    if step is None:
        return
    step_unit = row.get("step_unit")
    if iters_per_epoch is None or iters_per_epoch <= 0:
        return

    if step_unit == "iter":
        row["step_iter"] = int(step)
        row["step_epoch"] = float(step) / float(iters_per_epoch)
    elif step_unit == "epoch":
        row["step_epoch"] = float(step)
        row["step_iter"] = int(float(step) * float(iters_per_epoch))


def _write_outputs(out_dir: Path, rows: Iterable[Dict[str, Any]]) -> None:
    out_jsonl = out_dir / "metrics_std.jsonl"
    out_csv = out_dir / "metrics_std.csv"

    normalized: List[Dict[str, Any]] = []
    for r in rows:
        normalized.append({k: r.get(k, None) for k in FIELDS})

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in normalized:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in normalized:
            writer.writerow(r)


def _from_magformer(out_dir: Path, iters_per_epoch: Optional[int]) -> List[Dict[str, Any]]:
    rows_in = _load_jsonl(out_dir / "metrics_log.jsonl")
    out: List[Dict[str, Any]] = []
    for r in rows_in:
        it = _fint(r.get("iter"))
        phase = str(r.get("phase", ""))
        if it is None or phase not in {"train", "val"}:
            continue

        row = _blank()
        row["step"] = it
        row["step_unit"] = "iter"
        row["phase"] = phase
        row["wall_time_iso"] = r.get("wall_time_iso")
        row["elapsed_sec"] = _ffloat(r.get("elapsed_sec"))

        if phase == "train":
            row["train/loss"] = _ffloat(r.get("train/loss"))
            row["train/lr"] = _ffloat(r.get("train/lr"))
        else:
            for src, dst in [
                ("val/segm_AP", "val/segm_AP"),
                ("val/segm_AP50", "val/segm_AP50"),
                ("val/segm_AP75", "val/segm_AP75"),
                ("val/segm_APs", "val/segm_APs"),
                ("val/segm_APm", "val/segm_APm"),
                ("val/segm_APl", "val/segm_APl"),
                ("val/bbox_AP", "val/bbox_AP"),
                ("val/bbox_AP50", "val/bbox_AP50"),
                ("val/bbox_AP75", "val/bbox_AP75"),
                ("val/bbox_APs", "val/bbox_APs"),
                ("val/bbox_APm", "val/bbox_APm"),
                ("val/bbox_APl", "val/bbox_APl"),
            ]:
                row[dst] = _pct01_to_100(_ffloat(r.get(src)))

        _finalize_step_fields(row, iters_per_epoch)
        out.append(row)
    return out


def _from_detectron2(out_dir: Path, iters_per_epoch: Optional[int]) -> List[Dict[str, Any]]:
    rows_in = _load_jsonl(out_dir / "metrics.json")
    out: List[Dict[str, Any]] = []

    d2_to_std = {
        "segm/AP": "val/segm_AP",
        "segm/AP50": "val/segm_AP50",
        "segm/AP75": "val/segm_AP75",
        "segm/APs": "val/segm_APs",
        "segm/APm": "val/segm_APm",
        "segm/APl": "val/segm_APl",
        "bbox/AP": "val/bbox_AP",
        "bbox/AP50": "val/bbox_AP50",
        "bbox/AP75": "val/bbox_AP75",
        "bbox/APs": "val/bbox_APs",
        "bbox/APm": "val/bbox_APm",
        "bbox/APl": "val/bbox_APl",
    }

    for r in rows_in:
        it = _fint(r.get("iteration"))
        if it is None:
            continue

        is_val = any(k in r for k in ["segm/AP", "bbox/AP"])
        phase = "val" if is_val else "train"

        row = _blank()
        row["step"] = it
        row["step_unit"] = "iter"
        row["phase"] = phase
        # detectron2 metrics.json does not reliably include wall-time info
        row["wall_time_iso"] = None
        row["elapsed_sec"] = None

        if phase == "train":
            row["train/loss"] = _ffloat(r.get("total_loss"))
            row["train/lr"] = _ffloat(r.get("lr"))
        else:
            for src, dst in d2_to_std.items():
                row[dst] = _ffloat(r.get(src))

        _finalize_step_fields(row, iters_per_epoch)
        out.append(row)
    return out


def _from_yolo(out_dir: Path, iters_per_epoch: Optional[int]) -> List[Dict[str, Any]]:
    csv_path = out_dir / "train" / "results.csv"
    if not csv_path.exists():
        return []

    def _get_any(row: Dict[str, str], keys: List[str]) -> Optional[float]:
        for k in keys:
            if k not in row:
                continue
            if row[k] == "":
                continue
            v = _ffloat(row[k])
            if v is not None:
                return v
        return None

    out: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            epoch = _fint(r.get("epoch"))
            if epoch is None:
                continue

            # Train row (losses + lr)
            train_row = _blank()
            train_row["step"] = epoch
            train_row["step_unit"] = "epoch"
            train_row["phase"] = "train"
            loss_terms = [
                _get_any(r, ["train/box_loss"]),
                _get_any(r, ["train/seg_loss"]),
                _get_any(r, ["train/cls_loss"]),
                _get_any(r, ["train/dfl_loss"]),
            ]
            loss_sum = sum(v for v in loss_terms if v is not None) if any(v is not None for v in loss_terms) else None
            train_row["train/loss"] = loss_sum
            train_row["train/lr"] = _get_any(r, ["lr/pg0", "lr/pg1", "lr/pg2", "lr"])
            _finalize_step_fields(train_row, iters_per_epoch)
            out.append(train_row)

            # Val row (metrics)
            val_row = _blank()
            val_row["step"] = epoch
            val_row["step_unit"] = "epoch"
            val_row["phase"] = "val"
            val_row["val/segm_AP"] = _pct01_to_100(_get_any(r, ["metrics/mAP50-95(M)"]))
            val_row["val/segm_AP50"] = _pct01_to_100(_get_any(r, ["metrics/mAP50(M)"]))
            val_row["val/bbox_AP"] = _pct01_to_100(_get_any(r, ["metrics/mAP50-95(B)", "metrics/mAP50-95"]))
            val_row["val/bbox_AP50"] = _pct01_to_100(_get_any(r, ["metrics/mAP50(B)"]))
            _finalize_step_fields(val_row, iters_per_epoch)
            out.append(val_row)

    return out


def _from_cocoeval(out_dir: Path, iters_per_epoch: Optional[int]) -> List[Dict[str, Any]]:
    p = out_dir / "metrics.cocoeval.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    row = _blank()
    row["step"] = _fint(data.get("iteration", -1)) or -1
    row["step_unit"] = "iter"
    row["phase"] = "val"
    for src, dst in [
        ("segm/AP", "val/segm_AP"),
        ("segm/AP50", "val/segm_AP50"),
        ("segm/AP75", "val/segm_AP75"),
        ("segm/APs", "val/segm_APs"),
        ("segm/APm", "val/segm_APm"),
        ("segm/APl", "val/segm_APl"),
        ("bbox/AP", "val/bbox_AP"),
        ("bbox/AP50", "val/bbox_AP50"),
        ("bbox/AP75", "val/bbox_AP75"),
        ("bbox/APs", "val/bbox_APs"),
        ("bbox/APm", "val/bbox_APm"),
        ("bbox/APl", "val/bbox_APl"),
    ]:
        row[dst] = _ffloat(data.get(src))
    _finalize_step_fields(row, iters_per_epoch)
    return [row]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--framework", type=str, default="", choices=["", "magformer", "detectron2", "yolo", "cocoeval"])
    ap.add_argument("--iters-per-epoch", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    iters_per_epoch: Optional[int] = int(args.iters_per_epoch) if int(args.iters_per_epoch) > 0 else None

    framework = args.framework.strip() or _detect_framework(out_dir)
    if framework == "magformer":
        rows = _from_magformer(out_dir, iters_per_epoch)
    elif framework == "detectron2":
        rows = _from_detectron2(out_dir, iters_per_epoch)
    elif framework == "yolo":
        rows = _from_yolo(out_dir, iters_per_epoch)
    elif framework == "cocoeval":
        rows = _from_cocoeval(out_dir, iters_per_epoch)
    else:
        rows = _from_cocoeval(out_dir, iters_per_epoch)

    _write_outputs(out_dir, rows)
    print(f"[metrics-std] wrote {len(rows)} rows to {out_dir / 'metrics_std.jsonl'}")


if __name__ == "__main__":
    main()
