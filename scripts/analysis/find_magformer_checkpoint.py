#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def find_magformer_checkpoint(out_dir: Path) -> Path:
    if (out_dir / "model_best.pth").exists():
        return out_dir / "model_best.pth"
    ckpts = sorted(out_dir.glob("checkpoint_iter_*.pth"))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No MagFormer checkpoint found under {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    print(find_magformer_checkpoint(Path(args.out_dir).resolve()))


if __name__ == "__main__":
    main()
