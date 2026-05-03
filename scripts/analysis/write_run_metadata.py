#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _repo_root() -> Path:
    # <repo>/scripts/analysis/write_run_metadata.py
    return Path(__file__).resolve().parents[2]


def _git_hash(repo_root: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        sha = proc.stdout.strip()
        return sha if sha else None
    except Exception:
        return None


def _read_wall_time_sec(out_dir: Path) -> Optional[float]:
    p = out_dir / "wall_time_sec.txt"
    if not p.exists():
        return None
    try:
        return float(p.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _env_info() -> Dict[str, Any]:
    def _pkg_version(name: str) -> Optional[str]:
        try:
            return importlib_metadata.version(name)
        except Exception:
            return None

    info: Dict[str, Any] = {
        "python": {
            "version": sys.version.splitlines()[0],
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }

    # Avoid importing torch here: some environments crash due to MKL/OpenMP
    # loader conflicts. Package metadata is sufficient for reproducibility.
    info["packages"] = {
        "torch": _pkg_version("torch"),
        "detectron2": _pkg_version("detectron2"),
        "ultralytics": _pkg_version("ultralytics"),
        "timm": _pkg_version("timm"),
    }

    return info


def _load_existing(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["start", "end"], required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--track", type=str, default="")
    ap.add_argument("--register", type=str, default="")
    ap.add_argument("--dataset-root", type=str, default="")
    ap.add_argument("--model-id", type=str, default="")
    ap.add_argument("--candidate-id", type=str, default="")
    ap.add_argument("--run-tag", type=str, default="")
    ap.add_argument("--command", type=str, default="")
    ap.add_argument("--iters-per-epoch", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--ims-per-batch", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"

    meta = _load_existing(meta_path)
    if args.phase == "start":
        if not args.track or not args.register or not args.dataset_root or not args.model_id:
            raise SystemExit("--track/--register/--dataset-root/--model-id are required for --phase start")

        repo_root = _repo_root()
        meta.update(
            {
                "track": args.track,
                "register": args.register,
                "dataset_root": args.dataset_root,
                "model_id": args.model_id,
                "candidate_id": args.candidate_id or None,
                "run_tag": args.run_tag or None,
                "command": args.command or None,
                "git_hash": _git_hash(repo_root),
                "env": _env_info(),
                "start_time_iso": _now_iso(),
                "end_time_iso": None,
                "wall_time_sec": None,
                "budget": {
                    "iters_per_epoch": int(args.iters_per_epoch) if int(args.iters_per_epoch) > 0 else None,
                    "max_iter": int(args.max_iter) if int(args.max_iter) > 0 else None,
                    "epochs": int(args.epochs) if int(args.epochs) > 0 else None,
                    "ims_per_batch": int(args.ims_per_batch) if int(args.ims_per_batch) > 0 else None,
                },
            }
        )
    else:
        meta["end_time_iso"] = _now_iso()
        wall = _read_wall_time_sec(out_dir)
        if wall is not None:
            meta["wall_time_sec"] = wall

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[metadata] wrote: {meta_path}")


if __name__ == "__main__":
    main()
