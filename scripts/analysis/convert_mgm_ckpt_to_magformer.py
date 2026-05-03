#!/usr/bin/env python3
"""Convert MGM_Mask2Former checkpoint to MAGFormer format.

Key mappings:
  rgb_backbone.X           → rgb_backbone.model.X
  depth_backbone.*         → depth_backbone.model.* (ConvNeXt remapping)
  mgm.*                    → fusion.*
  sem_seg_head.pixel_decoder.* → pixel_decoder.* (adapter/layer renaming)
  sem_seg_head.predictor.* → decoder.*
  criterion.*              → skip
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _map_depth_backbone_key(rest: str) -> Optional[str]:
    """Map MGM depth_backbone.* → magformer depth_backbone.model.*"""
    # stem: downsample_layers.0.{0,1}.* → stem_{0,1}.*
    m = re.match(r"downsample_layers\.0\.(\d+)\.(.+)", rest)
    if m:
        return f"depth_backbone.model.stem_{m.group(1)}.{m.group(2)}"

    # downsample: downsample_layers.{1,2,3}.{0,1}.* → stages_{i}.downsample.{0,1}.*
    m = re.match(r"downsample_layers\.(\d+)\.(\d+)\.(.+)", rest)
    if m:
        stage = int(m.group(1))  # 1,2,3
        return f"depth_backbone.model.stages_{stage}.downsample.{m.group(2)}.{m.group(3)}"

    # blocks: stages.{i}.{j}.dwconv.* → stages_{i}.blocks.{j}.conv_dw.*
    m = re.match(r"stages\.(\d+)\.(\d+)\.dwconv\.(.+)", rest)
    if m:
        return f"depth_backbone.model.stages_{m.group(1)}.blocks.{m.group(2)}.conv_dw.{m.group(3)}"

    # blocks: stages.{i}.{j}.pwconv1.* → stages_{i}.blocks.{j}.mlp.fc1.*
    m = re.match(r"stages\.(\d+)\.(\d+)\.pwconv1\.(.+)", rest)
    if m:
        return f"depth_backbone.model.stages_{m.group(1)}.blocks.{m.group(2)}.mlp.fc1.{m.group(3)}"

    # blocks: stages.{i}.{j}.pwconv2.* → stages_{i}.blocks.{j}.mlp.fc2.*
    m = re.match(r"stages\.(\d+)\.(\d+)\.pwconv2\.(.+)", rest)
    if m:
        return f"depth_backbone.model.stages_{m.group(1)}.blocks.{m.group(2)}.mlp.fc2.{m.group(3)}"

    # blocks: stages.{i}.{j}.{norm,gamma} → stages_{i}.blocks.{j}.{norm,gamma}
    m = re.match(r"stages\.(\d+)\.(\d+)\.(.+)", rest)
    if m:
        return f"depth_backbone.model.stages_{m.group(1)}.blocks.{m.group(2)}.{m.group(3)}"

    return None

def _map_pixel_decoder_key(rest: str) -> Optional[str]:
    """Map sem_seg_head.pixel_decoder.* → pixel_decoder.*"""
    # adapter_N.weight → lateral_convs.(N-1).0.weight
    m = re.match(r"adapter_(\d+)\.weight$", rest)
    if m:
        idx = int(m.group(1)) - 1
        return f"pixel_decoder.lateral_convs.{idx}.0.weight"

    # adapter_N.norm.* → lateral_convs.(N-1).1.*
    m = re.match(r"adapter_(\d+)\.norm\.(.+)$", rest)
    if m:
        idx = int(m.group(1)) - 1
        return f"pixel_decoder.lateral_convs.{idx}.1.{m.group(2)}"

    # layer_N.weight → output_convs.(N-1).0.weight
    m = re.match(r"layer_(\d+)\.weight$", rest)
    if m:
        idx = int(m.group(1)) - 1
        return f"pixel_decoder.output_convs.{idx}.0.weight"

    # layer_N.norm.* → output_convs.(N-1).1.*
    m = re.match(r"layer_(\d+)\.norm\.(.+)$", rest)
    if m:
        idx = int(m.group(1)) - 1
        return f"pixel_decoder.output_convs.{idx}.1.{m.group(2)}"

    # Everything else: direct prefix swap
    return f"pixel_decoder.{rest}"


def map_key(k: str) -> Tuple[Optional[str], str]:
    """Map MGM key → MAGFormer key. Returns (mapped_key, reason)."""
    if k.startswith("criterion."):
        return None, "skip:criterion"

    # RGB backbone: add model. prefix
    if k.startswith("rgb_backbone."):
        rest = k[len("rgb_backbone."):]
        return f"rgb_backbone.model.{rest}", "ok:rgb"

    # Depth backbone: ConvNeXt remapping
    if k.startswith("depth_backbone."):
        rest = k[len("depth_backbone."):]
        mk = _map_depth_backbone_key(rest)
        if mk:
            return mk, "ok:depth"
        return None, "skip:depth_unmapped"

    # Fusion: mgm.* → fusion.*
    if k.startswith("mgm."):
        rest = k[len("mgm."):]
        return f"fusion.{rest}", "ok:fusion"

    # Pixel decoder
    if k.startswith("sem_seg_head.pixel_decoder."):
        rest = k[len("sem_seg_head.pixel_decoder."):]
        mk = _map_pixel_decoder_key(rest)
        if mk:
            return mk, "ok:pixel_decoder"
        return None, "skip:pixel_decoder_unmapped"

    # Decoder: predictor.static_query → decoder.query_feat
    if k == "sem_seg_head.predictor.static_query.weight":
        return "decoder.query_feat.weight", "ok:query_feat"

    if k.startswith("sem_seg_head.predictor."):
        rest = k[len("sem_seg_head.predictor."):]
        return f"decoder.{rest}", "ok:decoder"

    return None, "skip:unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert MGM checkpoint to MAGFormer format")
    ap.add_argument("--input", type=str, required=True, help="MGM model_final.pth path")
    ap.add_argument("--output", type=str, required=True, help="Output MAGFormer .pth path")
    ap.add_argument("--magformer-config", type=str, required=True, help="MAGFormer YAML config")
    ap.add_argument("--dataset-root", type=str, default="", help="Dataset root for config")
    args = ap.parse_args()

    from magformer.config import load_config
    from magformer.models import build_model

    # Load MGM checkpoint
    ckpt = torch.load(args.input, map_location="cpu")
    mgm_state = ckpt.get("model", ckpt)
    print(f"[convert-mgm] input={args.input} ({len(mgm_state)} keys)")

    # Build magformer model to get target shapes
    cfg_overrides: Dict[str, Any] = {}
    if args.dataset_root:
        cfg_overrides.setdefault("data", {})["dataset_root"] = args.dataset_root
    cfg = load_config(args.magformer_config, overrides=cfg_overrides)
    try:
        cfg.model.magformer.rgb_backbone.pretrained = False
        cfg.model.magformer.depth_backbone.pretrained = False
    except Exception:
        pass
    model = build_model(cfg)
    target_state = model.state_dict()

    # Map keys
    kept: Dict[str, torch.Tensor] = {}
    stats = {"ok": 0, "skip": 0, "shape_mismatch": 0, "missing_target": 0}

    for k, v in mgm_state.items():
        mk, reason = map_key(k)
        if mk is None:
            stats["skip"] += 1
            continue

        tv = v if torch.is_tensor(v) else torch.as_tensor(v)
        tgt = target_state.get(mk)
        if tgt is None:
            stats["missing_target"] += 1
            print(f"  WARN: {k} → {mk} not in target model")
            continue
        if tuple(tgt.shape) != tuple(tv.shape):
            stats["shape_mismatch"] += 1
            print(f"  WARN: {k} → {mk} shape {tuple(tv.shape)} != {tuple(tgt.shape)}")
            continue

        kept[mk] = tv
        stats["ok"] += 1

    # Save
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": kept, "source": args.input, "stats": stats}, out_path)

    print(f"[convert-mgm] output={out_path}")
    print(f"[convert-mgm] kept={stats['ok']} skip={stats['skip']} "
          f"shape_mismatch={stats['shape_mismatch']} missing={stats['missing_target']}")
    print(f"[convert-mgm] target has {len(target_state)} keys, mapped {stats['ok']}")


if __name__ == "__main__":
    main()
