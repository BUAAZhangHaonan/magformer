#!/usr/bin/env python3
"""Compare MagFormer and Mask2Former parameter distributions.

This script fixes a known grouping issue where MagFormer `decoder.*` parameters
were incorrectly counted as `other`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict


MAGFORMER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAGFORMER_ROOT.parent
MASK2FORMER_ROOT = PROJECT_ROOT / "mask2former" / "MGM_Mask2Former"

# Ensure both codebases are importable.
sys.path.insert(0, str(MAGFORMER_ROOT))
sys.path.insert(0, str(MASK2FORMER_ROOT))


def count_parameters(model: Any, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def classify_parameter_group(name: str) -> str:
    """Classify a parameter name into stable comparison groups.

    Priority matters. Decoder-specific rules run before generic head rules.
    """
    n = name.lower()

    if "rgb_backbone" in n or ".swin" in n:
        return "rgb_backbone"

    if "depth_backbone" in n or ".convnext" in n:
        return "depth_backbone"

    if "modality_fusion" in n or n.startswith("fusion.") or ".mgm" in n or n.startswith("mgm."):
        return "fusion/mgm"

    pixel_decoder_markers = (
        "pixel_decoder",
        "sem_seg_head.adapter_",
        "sem_seg_head.layer_",
        "sem_seg_head.input_proj",
        "sem_seg_head.transformer.encoder",
        "sem_seg_head.mask_features",
        "decoder.transformer.encoder",
        "decoder.input_proj",
        "decoder.lateral_convs",
        "decoder.output_convs",
        "decoder.mask_features",
    )
    if any(m in n for m in pixel_decoder_markers):
        return "pixel_decoder"

    transformer_decoder_markers = (
        "transformer_decoder",
        "sem_seg_head.predictor",
        "decoder.decoder",
        "decoder.class_embed",
        "decoder.mask_embed",
        "decoder.query_embed",
        "decoder.query_feat",
        "decoder.level_embed",
        "decoder.input_proj",
        "decoder.transformer_self_attention_layers",
        "decoder.transformer_cross_attention_layers",
        "decoder.transformer_ffn_layers",
    )
    if any(m in n for m in transformer_decoder_markers):
        return "transformer_decoder"

    if "class_embed" in n or "mask_embed" in n or "aux_outputs" in n:
        return "heads"

    return "other"


def get_parameter_groups(model: Any) -> Dict[str, int]:
    groups = {
        "rgb_backbone": 0,
        "depth_backbone": 0,
        "fusion/mgm": 0,
        "pixel_decoder": 0,
        "transformer_decoder": 0,
        "heads": 0,
        "other": 0,
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        g = classify_parameter_group(name)
        groups[g] += int(param.numel())
    return groups


def load_magformer_model(config_path: str):
    from magformer.config import load_config
    from magformer.models import build_model

    cfg = load_config(config_path, overrides={"data": {"dataset_root": "/tmp/dummy_dataset"}})
    model = build_model(cfg)
    return model


def load_mask2former_model(config_path: str):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.modeling import build_model
    from mask2former import add_maskformer2_config, add_mgm_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_mgm_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.freeze()
    model = build_model(cfg)
    return model


def _print_table_row(name: str, a: int, b: int) -> None:
    diff = a - b
    denom = max(a, b, 1)
    pct = abs(diff) / denom * 100.0
    print(f"{name:<22} {a:>14,} {b:>14,} {diff:>11,} {pct:>7.2f}%")


def run_comparison(magformer_config: Path, mask2former_config: Path) -> None:
    print("=" * 84)
    print("MagFormer vs Mask2Former parameter comparison")
    print("=" * 84)

    mag_model = load_magformer_model(str(magformer_config))
    m2f_model = load_mask2former_model(str(mask2former_config))

    mag_total = count_parameters(mag_model)
    m2f_total = count_parameters(m2f_model)
    mag_groups = get_parameter_groups(mag_model)
    m2f_groups = get_parameter_groups(m2f_model)

    print("\nTotal Parameters")
    print("-" * 84)
    _print_table_row("total", mag_total, m2f_total)

    print("\nGrouped Parameters")
    print("-" * 84)
    print(f"{'group':<22} {'MagFormer':>14} {'Mask2Former':>14} {'diff':>11} {'|diff|%':>8}")
    for k in [
        "rgb_backbone",
        "depth_backbone",
        "fusion/mgm",
        "pixel_decoder",
        "transformer_decoder",
        "heads",
        "other",
    ]:
        _print_table_row(k, mag_groups[k], m2f_groups[k])

    print("\nNotes")
    print("-" * 84)
    print("- `decoder.transformer.encoder.*` is counted as pixel_decoder (MagFormer).")
    print("- `sem_seg_head.predictor.*` is counted as transformer_decoder (Mask2Former).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model parameter grouping")
    parser.add_argument(
        "--magformer-config",
        default=str(MAGFORMER_ROOT / "configs" / "magformer_aligned_comparison.yaml"),
        help="Path to MagFormer config",
    )
    parser.add_argument(
        "--mask2former-config",
        default=str(MASK2FORMER_ROOT / "configs" / "mgm_aligned_comparison.yaml"),
        help="Path to Mask2Former config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_comparison(Path(args.magformer_config), Path(args.mask2former_config))


if __name__ == "__main__":
    main()
