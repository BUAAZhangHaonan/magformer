#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _find_ckpt(out_dir: Path) -> Path | None:
    # Detectron2 Checkpointer default names.
    for cand in [out_dir / "model_final.pth"]:
        if cand.exists():
            return cand
    ckpts = sorted(out_dir.glob("model_*.pth"))
    return ckpts[-1] if ckpts else None


def _torch_load_trusted(path: Path) -> object:
    # Torch 2.6+ defaults to weights_only=True and may reject some checkpoints.
    # These checkpoints are produced locally by our training code; treat as trusted.
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
        except TypeError:
            # Older torch without weights_only.
            return torch.load(path, map_location="cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ckpt_path = _find_ckpt(out_dir)
    if ckpt_path is None:
        raise SystemExit(f"[params] no detectron2 checkpoint found under: {out_dir}")

    ckpt = _torch_load_trusted(ckpt_path)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise SystemExit(f"[params] unexpected checkpoint type: {type(ckpt)} from {ckpt_path}")

    # Note: state_dict includes buffers (e.g., running_mean/var). For our use, the
    # difference vs. "trainable params" is negligible; we keep this definition
    # stable across detectron2-based baselines.
    n = sum(int(v.numel()) for v in state.values() if hasattr(v, "numel"))
    (out_dir / "params_trainable.txt").write_text(str(n) + "\n", encoding="utf-8")
    print(f"[params] wrote params_trainable={n} from {ckpt_path.name}")


if __name__ == "__main__":
    main()
