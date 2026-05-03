#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--magformer-metrics", type=str, required=True)
    ap.add_argument("--mask2former-metrics", type=str, required=True)
    ap.add_argument("--summary-file", type=str, required=True)
    args = ap.parse_args()

    mag_metrics_file = Path(args.magformer_metrics)
    m2f_metrics_file = Path(args.mask2former_metrics)
    summary_file = Path(args.summary_file)

    mag_rows = load_jsonl(mag_metrics_file)
    mag_val_rows = [r for r in mag_rows if r.get("phase") == "val" and "val/segm_AP" in r]
    if not mag_val_rows:
        raise RuntimeError(f"No val rows with segm_AP in {mag_metrics_file}")
    mag_last = mag_val_rows[-1]
    mag_best = max(mag_val_rows, key=lambda r: float(r.get("val/segm_AP", -1e9)))

    def to_pct(x: float) -> float:
        return float(x) * 100.0

    m2f_rows = load_jsonl(m2f_metrics_file)
    m2f_val_rows = [r for r in m2f_rows if "segm/AP" in r]
    if not m2f_val_rows:
        raise RuntimeError(f"No segm/AP rows in {m2f_metrics_file}")
    m2f_last = m2f_val_rows[-1]
    m2f_best = max(m2f_val_rows, key=lambda r: float(r.get("segm/AP", -1e9)))

    summary = {
        "track": "track_a_2k",
        "magformer": {
            "last_iter": int(mag_last.get("iter", -1)),
            "last": {
                "AP": to_pct(mag_last.get("val/segm_AP", 0.0)),
                "AP50": to_pct(mag_last.get("val/segm_AP50", 0.0)),
                "AP75": to_pct(mag_last.get("val/segm_AP75", 0.0)),
                "APs": to_pct(mag_last.get("val/segm_AP_small", 0.0)),
                "APm": to_pct(mag_last.get("val/segm_AP_medium", 0.0)),
                "APl": to_pct(mag_last.get("val/segm_AP_large", 0.0)),
            },
            "best": {
                "iter": int(mag_best.get("iter", -1)),
                "AP": to_pct(mag_best.get("val/segm_AP", 0.0)),
                "AP50": to_pct(mag_best.get("val/segm_AP50", 0.0)),
                "AP75": to_pct(mag_best.get("val/segm_AP75", 0.0)),
                "APs": to_pct(mag_best.get("val/segm_AP_small", 0.0)),
                "APm": to_pct(mag_best.get("val/segm_AP_medium", 0.0)),
                "APl": to_pct(mag_best.get("val/segm_AP_large", 0.0)),
            },
        },
        "mask2former": {
            "last_iter": int(m2f_last.get("iteration", -1)),
            "last": {
                "AP": float(m2f_last.get("segm/AP", 0.0)),
                "AP50": float(m2f_last.get("segm/AP50", 0.0)),
                "AP75": float(m2f_last.get("segm/AP75", 0.0)),
                "APs": float(m2f_last.get("segm/APs", 0.0)),
                "APm": float(m2f_last.get("segm/APm", 0.0)),
                "APl": float(m2f_last.get("segm/APl", 0.0)),
            },
            "best": {
                "iter": int(m2f_best.get("iteration", -1)),
                "AP": float(m2f_best.get("segm/AP", 0.0)),
                "AP50": float(m2f_best.get("segm/AP50", 0.0)),
                "AP75": float(m2f_best.get("segm/AP75", 0.0)),
                "APs": float(m2f_best.get("segm/APs", 0.0)),
                "APm": float(m2f_best.get("segm/APm", 0.0)),
                "APl": float(m2f_best.get("segm/APl", 0.0)),
            },
        },
    }

    summary["gap_last_ap"] = summary["magformer"]["last"]["AP"] - summary["mask2former"]["last"]["AP"]
    summary["gap_best_ap"] = summary["magformer"]["best"]["AP"] - summary["mask2former"]["best"]["AP"]

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[summary] wrote: {summary_file}")


if __name__ == "__main__":
    main()
