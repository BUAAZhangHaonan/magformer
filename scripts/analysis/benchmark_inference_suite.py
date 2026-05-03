#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "analysis" / "benchmark_inference.py"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _auto_scan_models(output_root: Path) -> List[Path]:
    ignore = {"_tuning", "visualizations"}
    out: List[Path] = []
    if not output_root.exists():
        return out
    for path in sorted(output_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name in ignore or path.name.startswith(".") or path.name.startswith("_"):
            continue
        out.append(path)
    return out


def _collect_model_dirs(output_root: Path, summary_path: Path | None) -> List[Path]:
    if summary_path is None:
        return _auto_scan_models(output_root)

    payload = _load_json(summary_path)
    out: List[Path] = []
    for model_id, entry in payload.items():
        if model_id in {"experiment", "output_root"} or not isinstance(entry, dict):
            continue
        path = output_root / model_id
        if path.exists():
            out.append(path)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference for all model outputs under a suite root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timed-images", type=int, default=50)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    summary_path = Path(args.summary).resolve() if args.summary else None

    model_dirs = _collect_model_dirs(output_root, summary_path)
    seen_realpaths: Set[Path] = set()
    filtered_dirs: List[Path] = []
    for model_dir in model_dirs:
        realpath = model_dir.resolve()
        if realpath in seen_realpaths:
            continue
        seen_realpaths.add(realpath)
        filtered_dirs.append(model_dir)

    for model_dir in filtered_dirs:
        cmd = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--out-dir",
            str(model_dir),
            "--dataset-root",
            str(dataset_root),
            "--device",
            str(args.device),
            "--warmup",
            str(int(args.warmup)),
            "--timed-images",
            str(int(args.timed_images)),
        ]
        print(f"[benchmark-suite] {model_dir.name}")
        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0 and not args.continue_on_error:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
