#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_iters_per_epoch(output_root: Path, model_id: str) -> Optional[int]:
    metadata_path = output_root / model_id / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = _load_json(metadata_path)
    except Exception:
        return None
    budget = metadata.get("budget", {})
    try:
        return int(budget.get("iters_per_epoch"))
    except Exception:
        return None


def _pack_entry(entry: Dict[str, Any], *, iters_per_epoch: Optional[int]) -> Dict[str, Any]:
    best = entry.get("best", {})
    last = entry.get("last", {})
    best_iter = best.get("iter", best.get("epoch", -1))
    last_iter = last.get("iter", last.get("epoch", -1))

    def _epoch(iter_or_epoch: Any) -> Optional[float]:
        if iter_or_epoch is None:
            return None
        try:
            value = float(iter_or_epoch)
        except Exception:
            return None
        if iters_per_epoch and int(value) >= 0:
            return round(value / float(iters_per_epoch), 3)
        return value

    return {
        "best_ap": ((best.get("segm") or {}).get("AP")),
        "last_ap": ((last.get("segm") or {}).get("AP")),
        "best_iter": best_iter,
        "last_iter": last_iter,
        "best_epoch": _epoch(best_iter),
        "last_epoch": _epoch(last_iter),
    }


def build_comparison(summary: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    pairs = {
        "magformer": {
            "control": "magformer_nodpth_ref",
            "depth_on": "magformer_depthnorm_on",
        },
        "mgm_mask2former": {
            "control": "mgm_mask2former_nodpth_ref",
            "depth_on": "mgm_mask2former_depthnorm_on",
        },
    }

    result: Dict[str, Any] = {
        "experiment": summary.get("experiment"),
        "output_root": str(output_root),
        "pairs": {},
    }
    for family, mapping in pairs.items():
        pair_result: Dict[str, Any] = {}
        for phase, model_id in mapping.items():
            entry = summary.get(model_id, {})
            pair_result[phase] = _pack_entry(
                entry,
                iters_per_epoch=_read_iters_per_epoch(output_root, model_id),
            )
            pair_result[phase]["model_id"] = model_id
        result["pairs"][family] = pair_result
    return result


def write_csv(comparison: Dict[str, Any], path: Path) -> None:
    rows = []
    for family, pair in (comparison.get("pairs") or {}).items():
        for phase, entry in pair.items():
            rows.append(
                {
                    "family": family,
                    "phase": phase,
                    "model_id": entry.get("model_id"),
                    "best_ap": entry.get("best_ap"),
                    "last_ap": entry.get("last_ap"),
                    "best_iter": entry.get("best_iter"),
                    "last_iter": entry.get("last_iter"),
                    "best_epoch": entry.get("best_epoch"),
                    "last_epoch": entry.get("last_epoch"),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "phase",
                "model_id",
                "best_ap",
                "last_ap",
                "best_iter",
                "last_iter",
                "best_epoch",
                "last_epoch",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, required=True)
    ap.add_argument("--output-json", type=str, default="")
    ap.add_argument("--output-csv", type=str, default="")
    args = ap.parse_args()

    summary_path = Path(args.summary).resolve()
    summary = _load_json(summary_path)
    output_root = Path(summary["output_root"]).resolve()
    comparison = build_comparison(summary, output_root)

    output_json = Path(args.output_json).resolve() if args.output_json else output_root / "revisit_comparison.json"
    output_csv = Path(args.output_csv).resolve() if args.output_csv else output_root / "revisit_comparison.csv"

    output_json.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(comparison, output_csv)
    print(f"[revisit-comparison] wrote json: {output_json}")
    print(f"[revisit-comparison] wrote csv: {output_csv}")


if __name__ == "__main__":
    main()
