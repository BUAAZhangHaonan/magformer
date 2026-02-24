#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

# File: <repo>/scripts/analysis/convert_mask2former_ckpt_to_magformer.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from magformer.config import load_config  # noqa: E402
from magformer.models import build_model  # noqa: E402


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
    model = obj.get("model", None)
    if not isinstance(model, dict):
        raise ValueError(f"Invalid checkpoint: missing dict 'model' in {path}")
    return model


def _reduce_coco_class_embed_weight(weight: torch.Tensor) -> torch.Tensor:
    """
    Reduce a COCO-style class embedding weight of shape (K+1, D) to 1-class shape (2, D).

    Strategy:
      - foreground row = mean over all foreground classes
      - no-object row = keep the checkpoint's last row
    """
    if weight.ndim != 2 or weight.shape[0] <= 2:
        return weight
    fg = weight[:-1].mean(dim=0, keepdim=True)
    no = weight[-1:].clone()
    return torch.cat([fg, no], dim=0)


def _reduce_coco_class_embed_bias(bias: torch.Tensor) -> torch.Tensor:
    """
    Reduce a COCO-style class embedding bias of shape (K+1,) to 1-class shape (2,).

    Mirrors `_reduce_coco_class_embed_weight`.
    """
    if bias.ndim != 1 or bias.shape[0] <= 2:
        return bias
    fg = bias[:-1].mean(dim=0, keepdim=True)
    no = bias[-1:].clone()
    return torch.cat([fg, no], dim=0)


def _map_key(k: str) -> Tuple[str | None, str]:
    """
    Map Mask2Former detectron2-style keys to MAGFormer keys.

    Returns:
        (mapped_key_or_none, reason)
    """
    if k.startswith("criterion."):
        return None, "skip:criterion"

    # Mask2Former uses `static_query` for learnable query features.
    # Our implementation uses `query_feat`.
    if k == "sem_seg_head.predictor.static_query.weight":
        return "decoder.query_feat.weight", "ok:query_feat"

    if k.startswith("sem_seg_head.predictor.class_embed."):
        # Convert COCO (80+1) -> ECC (1+1) by reducing to 2 rows in main().
        rest = k[len("sem_seg_head.predictor.class_embed.") :]
        return f"decoder.class_embed.{rest}", "ok:class_embed"

    if k.startswith("backbone."):
        rest = k[len("backbone.") :]
        # timm features_only wrapper uses module names like layers_0 instead of layers.0
        rest = re.sub(r"\blayers\.(\d+)\.", r"layers_\1.", rest)
        # Our Swin wrapper is `rgb_backbone.model` (FeatureListNet / timm model).
        return f"rgb_backbone.model.{rest}", "ok:backbone"

    # Detectron2 MSDeformAttnPixelDecoder uses adapter_N/layer_N naming for FPN levels.
    # Our implementation uses `lateral_convs`/`output_convs` with (Conv2d, GN) in a Sequential.
    m = re.match(r"^sem_seg_head\.pixel_decoder\.adapter_(\d+)\.(.+)$", k)
    if m is not None:
        level = int(m.group(1))
        tail = m.group(2)
        idx = level - 1
        if tail == "weight":
            return f"pixel_decoder.lateral_convs.{idx}.0.weight", "ok:pixel_decoder_adapter"
        if tail.startswith("norm."):
            return f"pixel_decoder.lateral_convs.{idx}.1.{tail[len('norm.'):]}", "ok:pixel_decoder_adapter"

    m = re.match(r"^sem_seg_head\.pixel_decoder\.layer_(\d+)\.(.+)$", k)
    if m is not None:
        level = int(m.group(1))
        tail = m.group(2)
        idx = level - 1
        if tail == "weight":
            return f"pixel_decoder.output_convs.{idx}.0.weight", "ok:pixel_decoder_layer"
        if tail.startswith("norm."):
            return f"pixel_decoder.output_convs.{idx}.1.{tail[len('norm.'):]}", "ok:pixel_decoder_layer"

    if k.startswith("sem_seg_head.pixel_decoder."):
        rest = k[len("sem_seg_head.pixel_decoder.") :]
        return f"pixel_decoder.{rest}", "ok:pixel_decoder"

    if k.startswith("sem_seg_head.predictor."):
        rest = k[len("sem_seg_head.predictor.") :]
        return f"decoder.{rest}", "ok:predictor"

    return None, "skip:unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=DEFAULT_URL, help="Mask2Former .pkl path or URL")
    ap.add_argument("--output", type=str, required=True, help="Output MagFormer warm-start .pth")
    ap.add_argument(
        "--magformer-config",
        type=str,
        required=True,
        help="MAGFormer YAML config (used to filter keys by existence/shape)",
    )
    ap.add_argument("--dataset-root", type=str, default="", help="Optional dataset root override for config")
    ap.add_argument(
        "--include-class-embed",
        action="store_true",
        help="Also warm-start `decoder.class_embed.*` by reducing COCO (80+1) -> ECC (1+1).",
    )
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

    mask2former_state = _load_mask2former_pkl(ckpt_path)

    cfg_overrides: Dict[str, Any] = {}
    if args.dataset_root:
        cfg_overrides.setdefault("data", {})["dataset_root"] = args.dataset_root
    cfg = load_config(args.magformer_config, overrides=cfg_overrides)
    # Conversion only needs tensor shapes; avoid downloading any pretrained weights.
    try:
        if getattr(getattr(cfg, "model", None), "magformer", None) is not None:
            cfg.model.magformer.rgb_backbone.pretrained = False
            cfg.model.magformer.depth_backbone.pretrained = False
    except Exception:
        pass
    model = build_model(cfg)
    target_state = model.state_dict()

    kept: Dict[str, torch.Tensor] = {}
    skipped = {
        "skip:criterion": 0,
        "skip:class_embed": 0,
        "skip:unknown": 0,
        "skip:missing": 0,
        "skip:shape": 0,
    }

    for k, v in mask2former_state.items():
        if k.startswith("sem_seg_head.predictor.class_embed.") and not args.include_class_embed:
            skipped["skip:class_embed"] += 1
            continue
        mk, reason = _map_key(k)
        if mk is None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        tv = _to_tensor(v)
        if k == "sem_seg_head.predictor.class_embed.weight":
            tv = _reduce_coco_class_embed_weight(tv)
        elif k == "sem_seg_head.predictor.class_embed.bias":
            tv = _reduce_coco_class_embed_bias(tv)
        tgt = target_state.get(mk, None)
        if tgt is None:
            skipped["skip:missing"] += 1
            continue
        if tuple(tgt.shape) != tuple(tv.shape):
            skipped["skip:shape"] += 1
            continue

        kept[mk] = tv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": kept,
        "source": {
            "input": src,
            "resolved_path": str(ckpt_path),
        },
        "stats": {
            "kept": len(kept),
            **skipped,
        },
    }
    torch.save(payload, out_path)

    print(f"[convert-mask2former] input={src}")
    print(f"[convert-mask2former] resolved={ckpt_path}")
    print(f"[convert-mask2former] output={out_path}")
    print(f"[convert-mask2former] kept={len(kept)} skipped={skipped}")


if __name__ == "__main__":
    main()
