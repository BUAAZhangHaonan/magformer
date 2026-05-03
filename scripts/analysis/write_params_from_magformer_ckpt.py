#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _torch_load_trusted(path: Path) -> object:
    # Torch 2.6+ defaults to weights_only=True and may reject some checkpoints.
    # These checkpoints are produced locally by our training code; treat as trusted.
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
        except TypeError:
            return torch.load(path, map_location="cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ckpts = sorted(out_dir.glob("checkpoint_iter_*.pth"))
    if not ckpts:
        raise SystemExit(f"[params] no checkpoint_iter_*.pth found under: {out_dir}")

    ckpt_path = ckpts[-1]
    ckpt = _torch_load_trusted(ckpt_path)

    if isinstance(ckpt, dict):
        # MagFormer checkpoints store the model weights under this key.
        state = (
            ckpt.get("model_state_dict")
            or ckpt.get("model")
            or ckpt.get("state_dict")
            or ckpt
        )
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise SystemExit(f"[params] unexpected checkpoint type: {type(ckpt)} from {ckpt_path}")

    n = sum(int(v.numel()) for v in state.values() if hasattr(v, "numel"))
    (out_dir / "params_trainable.txt").write_text(str(n) + "\n", encoding="utf-8")
    print(f"[params] wrote params_trainable={n} from {ckpt_path.name}")


if __name__ == "__main__":
    main()
