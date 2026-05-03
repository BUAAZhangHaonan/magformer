#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# File: <repo>/scripts/experiments/postprocess_cocoeval.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.coco_eval_results import evaluate_coco_results
from baselines.coco_eval_results import _normalize_results_rows


def _find_results_json(out_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Results json not found: {explicit}")
        return explicit

    candidates = [
        out_dir / "coco_instances_results.json",
        out_dir / "inference" / "coco_instances_results.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Cannot find coco results json under {out_dir}. Tried: {', '.join(str(c) for c in candidates)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--ann-file", type=str, default="")
    ap.add_argument("--results-json", type=str, default="")
    ap.add_argument("--metrics-out", type=str, default="metrics.cocoeval.json")
    ap.add_argument("--iteration", type=int, default=-1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    ann_file = Path(args.ann_file).resolve() if args.ann_file else (dataset_root / "annotations" / "instances_val.json")
    explicit_results = Path(args.results_json).resolve() if args.results_json else None

    out_dir.mkdir(parents=True, exist_ok=True)

    results_json = _find_results_json(out_dir=out_dir, explicit=explicit_results)
    canonical_results = out_dir / "coco_instances_results.json"
    if results_json.resolve() != canonical_results.resolve():
        shutil.copy2(results_json, canonical_results)

    rows = json.loads(canonical_results.read_text(encoding="utf-8"))
    normalized_rows, changed = _normalize_results_rows(rows)
    if changed:
        canonical_results.write_text(
            json.dumps(normalized_rows, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    metrics = evaluate_coco_results(
        ann_file=ann_file,
        results_json=canonical_results,
        iteration=int(args.iteration),
    )

    metrics_out = Path(args.metrics_out)
    if not metrics_out.is_absolute():
        metrics_out = out_dir / metrics_out
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[postprocess-cocoeval] results={canonical_results}")
    print(f"[postprocess-cocoeval] metrics={metrics_out}")


if __name__ == "__main__":
    main()
