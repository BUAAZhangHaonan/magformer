#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _read_metrics(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_wall_time(path: Path) -> float:
    if not path.exists():
        return float("inf")
    return _safe_float(path.read_text(encoding="utf-8").strip(), float("inf"))


def _candidate_key(item: Dict[str, Any], priority: Dict[str, int]) -> Tuple[float, float, int]:
    # Higher segm/AP is better, lower wall time is better, lower priority index is better.
    return (
        _safe_float(item.get("segm_ap"), float("-inf")),
        -_safe_float(item.get("wall_time_sec"), float("inf")),
        -priority.get(str(item.get("candidate_id")), -999999),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=str, required=True)
    ap.add_argument("--model-id", type=str, required=True)
    ap.add_argument("--candidates", type=str, default="C1,C2,C3,C4")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    output_root = Path(args.output_root).resolve()
    model_id = str(args.model_id)
    candidate_ids = [x.strip() for x in args.candidates.split(",") if x.strip()]
    priority = {cid: idx for idx, cid in enumerate(candidate_ids)}

    rows: List[Dict[str, Any]] = []
    for cid in candidate_ids:
        cdir = output_root / "_tuning" / model_id / cid
        metrics = _read_metrics(cdir / "metrics.cocoeval.json")
        segm_ap = None if metrics is None else _safe_float(metrics.get("segm/AP"), float("-inf"))
        wall_time = _read_wall_time(cdir / "wall_time_sec.txt")
        rows.append(
            {
                "candidate_id": cid,
                "path": str(cdir),
                "status": "ok" if metrics is not None else "missing",
                "segm_ap": segm_ap,
                "wall_time_sec": wall_time,
            }
        )

    valid = [r for r in rows if r["status"] == "ok"]
    if not valid:
        raise SystemExit(f"No valid candidates found for {model_id} under {output_root}/_tuning/{model_id}")

    best = sorted(valid, key=lambda r: _candidate_key(r, priority), reverse=True)[0]
    payload = {"model_id": model_id, "best_candidate": best["candidate_id"], "candidates": rows}

    if args.write:
        out = output_root / "_tuning" / model_id / "best_candidate.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[select-best] wrote: {out}")

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
