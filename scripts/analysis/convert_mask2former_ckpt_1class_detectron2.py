#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
import urllib.request
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import torch

# File: <repo>/scripts/analysis/convert_mask2former_ckpt_1class_detectron2.py
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_URL = (
    "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/"
    "maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl"
)


def _to_tensor(v: Any) -> torch.Tensor:
    if torch.is_tensor(v):
        return v
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    return torch.as_tensor(v)


def _download_if_needed(src: str, dst: Path) -> Path:
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(src, dst)
    return dst


def _load_mask2former_pkl(path: Path) -> Dict[str, Any]:
    obj = pickle.load(open(path, "rb"))
    if not isinstance(obj, dict) or not isinstance(obj.get("model"), dict):
        raise ValueError(f"Invalid checkpoint: expected dict with 'model' in {path}")
    return obj


def _reduce_coco_linear_weight_to_1class(weight: torch.Tensor) -> torch.Tensor:
    """
    Reduce COCO class head weight from (K+1, D) to (2, D) for single-class training.
    Foreground row = mean of all foreground classes; no-object row = keep last row.
    """
    if weight.ndim != 2 or weight.shape[0] <= 2:
        return weight
    fg = weight[:-1].mean(dim=0, keepdim=True)
    no = weight[-1:].clone()
    return torch.cat([fg, no], dim=0)


def _reduce_coco_linear_bias_to_1class(bias: torch.Tensor) -> torch.Tensor:
    if bias.ndim != 1 or bias.shape[0] <= 2:
        return bias
    fg = bias[:-1].mean(dim=0, keepdim=True)
    no = bias[-1:].clone()
    return torch.cat([fg, no], dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=DEFAULT_URL, help="Mask2Former .pkl path or URL")
    ap.add_argument("--output", type=str, required=True, help="Output detectron2-style .pth")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    src = str(args.input)
    out_path = Path(args.output).resolve()

    if src.startswith("http://") or src.startswith("https://"):
        cache_dir = REPO_ROOT / "output" / "pretrained"
        cache_path = cache_dir / Path(src).name
        if args.force_download and cache_path.exists():
            cache_path.unlink()
        ckpt_path = _download_if_needed(src, cache_path)
    else:
        ckpt_path = Path(src).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Input checkpoint not found: {ckpt_path}")

    obj = _load_mask2former_pkl(ckpt_path)
    state_in = obj["model"]
    # Preserve detectron2 state_dict metadata so Mask2Former doesn't apply legacy key conversion
    # (e.g. sem_seg_head.* -> sem_seg_head.pixel_decoder.*) on our already-v2 keys.
    state: "OrderedDict[str, torch.Tensor]" = OrderedDict((k, _to_tensor(v)) for k, v in state_in.items())
    if hasattr(state_in, "_metadata"):
        state._metadata = getattr(state_in, "_metadata")  # type: ignore[attr-defined]

    # Reduce class head + criterion weights for NUM_CLASSES=1 (=> 2 logits incl. no-object).
    if "sem_seg_head.predictor.class_embed.weight" in state:
        state["sem_seg_head.predictor.class_embed.weight"] = _reduce_coco_linear_weight_to_1class(
            state["sem_seg_head.predictor.class_embed.weight"]
        )
    if "sem_seg_head.predictor.class_embed.bias" in state:
        state["sem_seg_head.predictor.class_embed.bias"] = _reduce_coco_linear_bias_to_1class(
            state["sem_seg_head.predictor.class_embed.bias"]
        )
    if "criterion.empty_weight" in state:
        state["criterion.empty_weight"] = _reduce_coco_linear_bias_to_1class(state["criterion.empty_weight"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state,
            "__author__": obj.get("__author__", ""),
            "source": {"input": src, "resolved_path": str(ckpt_path)},
        },
        out_path,
    )

    print(f"[convert-mask2former-1class] input={src}")
    print(f"[convert-mask2former-1class] resolved={ckpt_path}")
    print(f"[convert-mask2former-1class] output={out_path}")


if __name__ == "__main__":
    main()
