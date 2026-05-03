#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _find_weights(out_dir: Path) -> Path | None:
    weights = out_dir / "train" / "weights"
    best = weights / "best.pt"
    last = weights / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to best.pt/last.pt. If empty, auto-detect under OUT/train/weights.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pt = Path(args.weights) if args.weights else _find_weights(out_dir)
    if pt is None or not pt.exists():
        raise SystemExit(f"[yolo-params] weights not found under: {out_dir / 'train' / 'weights'}")

    # Import lazily to avoid importing ultralytics unless needed.
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(pt))
    # YOLO checkpoints loaded for inference usually have requires_grad=False for
    # all params. Count all parameters to keep parity with our other param
    # writers, which read the full state dict size from checkpoints.
    n = sum(int(p.numel()) for p in model.model.parameters())
    (out_dir / "params_trainable.txt").write_text(str(n) + "\n", encoding="utf-8")
    print(f"[yolo-params] wrote params_trainable={n} from {pt.name}")


if __name__ == "__main__":
    main()
