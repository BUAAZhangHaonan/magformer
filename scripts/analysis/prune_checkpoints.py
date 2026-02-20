#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _best_detectron2_iter(out_dir: Path) -> Optional[int]:
    rows = _load_jsonl(out_dir / "metrics.json")
    val = [r for r in rows if "segm/AP" in r and "iteration" in r]
    if not val:
        return None
    best = max(val, key=lambda r: float(r.get("segm/AP", -1e9)))
    return int(best["iteration"])


def _best_magformer_iter(out_dir: Path) -> Optional[int]:
    rows = _load_jsonl(out_dir / "metrics_log.jsonl")
    val = [r for r in rows if r.get("phase") == "val" and "val/segm_AP" in r]
    if not val:
        return None
    best = max(val, key=lambda r: float(r.get("val/segm_AP", -1e9)))
    return int(best["iter"])


def _nearest_detectron2_ckpt(out_dir: Path, target_iter: int) -> Optional[Path]:
    candidates = sorted(out_dir.glob("model_*.pth"))
    if not candidates:
        return None
    exact = out_dir / f"model_{target_iter:07d}.pth"
    if exact.exists():
        return exact
    plus_one = out_dir / f"model_{target_iter + 1:07d}.pth"
    if plus_one.exists():
        return plus_one

    parsed = []
    for p in candidates:
        m = re.match(r"model_(\d+)\.pth$", p.name)
        if m:
            parsed.append((abs(int(m.group(1)) - target_iter), p))
    if not parsed:
        return None
    parsed.sort(key=lambda x: x[0])
    return parsed[0][1]


def _rm(paths: Iterable[Path], dry_run: bool) -> int:
    removed = 0
    for p in paths:
        if not p.exists():
            continue
        print(f"[prune] remove: {p}")
        removed += 1
        if not dry_run:
            p.unlink()
    return removed


def _prune_detectron2(out_dir: Path, dry_run: bool) -> int:
    final_ckpt = out_dir / "model_final.pth"
    best_iter = _best_detectron2_iter(out_dir)
    best_ckpt = _nearest_detectron2_ckpt(out_dir, best_iter) if best_iter is not None else None

    keep = {final_ckpt.resolve()} if final_ckpt.exists() else set()
    if best_ckpt is not None and best_ckpt.exists():
        keep.add(best_ckpt.resolve())

    to_remove = []
    for ckpt in sorted(out_dir.glob("model_*.pth")):
        if ckpt.resolve() not in keep:
            to_remove.append(ckpt)

    print(f"[prune] framework=detectron2 best_iter={best_iter} best_ckpt={best_ckpt}")
    return _rm(to_remove, dry_run=dry_run)


def _prune_magformer(out_dir: Path, dry_run: bool) -> int:
    ckpts = sorted(out_dir.glob("checkpoint_iter_*.pth"))
    if not ckpts:
        return 0

    final_ckpt = ckpts[-1]
    best_iter = _best_magformer_iter(out_dir)
    best_ckpt = out_dir / f"checkpoint_iter_{best_iter:07d}.pth" if best_iter is not None else None
    model_best = out_dir / "model_best.pth"

    keep = {final_ckpt.resolve()}
    if best_ckpt is not None and best_ckpt.exists():
        keep.add(best_ckpt.resolve())
    if model_best.exists():
        keep.add(model_best.resolve())

    to_remove = [p for p in ckpts if p.resolve() not in keep]
    print(f"[prune] framework=magformer best_iter={best_iter} final_ckpt={final_ckpt.name}")
    return _rm(to_remove, dry_run=dry_run)


def _prune_yolo(out_dir: Path, dry_run: bool) -> int:
    weights_dir = out_dir / "train" / "weights"
    if not weights_dir.exists():
        return 0
    keep_names = {"best.pt", "last.pt"}
    to_remove = [p for p in sorted(weights_dir.glob("*.pt")) if p.name not in keep_names]
    print(f"[prune] framework=yolo keep={sorted(keep_names)}")
    return _rm(to_remove, dry_run=dry_run)


def _detect_framework(out_dir: Path) -> str:
    if (out_dir / "train" / "weights").exists():
        return "yolo"
    if any(out_dir.glob("checkpoint_iter_*.pth")) or (out_dir / "model_best.pth").exists():
        return "magformer"
    if (out_dir / "model_final.pth").exists() or any(out_dir.glob("model_*.pth")):
        return "detectron2"
    return "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument(
        "--framework",
        type=str,
        default="auto",
        choices=["auto", "detectron2", "magformer", "yolo", "ucn", "none"],
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    framework = args.framework if args.framework != "auto" else _detect_framework(out_dir)

    print(f"[prune] out_dir={out_dir}")
    print(f"[prune] framework={framework} mode={'dry-run' if args.dry_run else 'write'}")

    removed = 0
    if framework == "detectron2":
        removed = _prune_detectron2(out_dir, dry_run=args.dry_run)
    elif framework == "magformer":
        removed = _prune_magformer(out_dir, dry_run=args.dry_run)
    elif framework == "yolo":
        removed = _prune_yolo(out_dir, dry_run=args.dry_run)
    elif framework in {"ucn", "none"}:
        removed = 0
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    print(f"[prune] removed={removed}")


if __name__ == "__main__":
    main()
