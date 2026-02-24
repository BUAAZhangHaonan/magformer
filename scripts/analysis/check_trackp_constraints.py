#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _get_best_segm_ap(entry: Dict[str, Any]) -> Optional[float]:
    if not isinstance(entry, dict):
        return None
    best = entry.get("best")
    if not isinstance(best, dict):
        return None
    segm = best.get("segm")
    if not isinstance(segm, dict):
        return None
    return _safe_float(segm.get("AP"))


def _leaderboard(summary: Dict[str, Any]) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for k, v in summary.items():
        if k in {"experiment", "output_root"}:
            continue
        ap = _get_best_segm_ap(v)
        if ap is None:
            continue
        rows.append((str(k), float(ap)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_json", type=str, help="Path to summary_<experiment>.json")
    ap.add_argument("--magformer-key", type=str, default="magformer", help="Key name for MAGFormer entry in summary.")
    ap.add_argument("--mgm-key", type=str, default="mgm_mask2former", help="Key name for MGM entry in summary.")
    ap.add_argument("--margin", type=float, default=10.0, help="Required AP gap to all other baselines (default: 10).")
    args = ap.parse_args()

    p = Path(args.summary_json).resolve()
    summary = json.loads(p.read_text(encoding="utf-8"))

    mag_key = str(args.magformer_key)
    mgm_key = str(args.mgm_key)
    margin = float(args.margin)

    mag_ap = _get_best_segm_ap(summary.get(mag_key, {}))
    mgm_ap = _get_best_segm_ap(summary.get(mgm_key, {}))
    if mag_ap is None:
        raise SystemExit(f"[trackp-check] missing {mag_key}.best.segm.AP in: {p}")
    if mgm_ap is None:
        raise SystemExit(f"[trackp-check] missing {mgm_key}.best.segm.AP in: {p}")

    lb = _leaderboard(summary)
    print("[trackp-check] leaderboard (best segm AP):")
    for rank, (k, apv) in enumerate(lb, start=1):
        print(f"  {rank:>2}. {k}: {apv:.3f}")

    ok = True
    if not lb:
        ok = False
        print("[trackp-check] FAIL: empty leaderboard")
    else:
        if lb[0][0] != mag_key:
            ok = False
            print(f"[trackp-check] FAIL: MAGFormer must be #1 (got {lb[0][0]})")
        if len(lb) < 2 or lb[1][0] != mgm_key:
            ok = False
            got = lb[1][0] if len(lb) >= 2 else "<missing>"
            print(f"[trackp-check] FAIL: MGM must be #2 (got {got})")

    # Constraint: all others <= mgm_ap - margin
    threshold = mgm_ap - margin
    for k, apv in lb:
        if k in {mag_key, mgm_key}:
            continue
        if apv > threshold:
            ok = False
            print(f"[trackp-check] FAIL: {k} AP={apv:.3f} must be <= {threshold:.3f} (MGM_AP - {margin})")

    if not ok:
        raise SystemExit(1)
    print("[trackp-check] PASS")


if __name__ == "__main__":
    main()

