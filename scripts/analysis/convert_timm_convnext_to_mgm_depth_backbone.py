#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch


def map_timm_convnext_state_dict_to_mgm_depth_backbone(
    timm_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Convert timm ConvNeXt keys to MGM ConvNeXtDepthBackbone keys.

    timm uses:
      - stem.{0,1}.*
      - stages.{0..3}.downsample.{0,1}.*
      - stages.{0..3}.blocks.{b}.(conv_dw|norm|mlp.fc1|mlp.fc2|gamma).*

    MGM ConvNeXtDepthBackbone expects:
      - downsample_layers.{0..3}.{0,1}.*
      - stages.{0..3}.{b}.(dwconv|norm|pwconv1|pwconv2|gamma).*
    """

    out: Dict[str, torch.Tensor] = {}

    def put(k: str, v: torch.Tensor) -> None:
        if k in out:
            raise ValueError(f"Duplicate mapped key: {k}")
        out[k] = v

    for k, v in timm_state.items():
        # Stem
        if k.startswith("stem."):
            rest = k[len("stem.") :]
            if rest.startswith("0."):
                put("downsample_layers.0.0." + rest[len("0.") :], v)
                continue
            if rest.startswith("1."):
                put("downsample_layers.0.1." + rest[len("1.") :], v)
                continue

        # Downsample blocks between stages
        if k.startswith("stages.") and ".downsample." in k:
            parts = k.split(".")
            # stages.{stage}.downsample.{idx}.{param}
            if len(parts) >= 5 and parts[2] == "downsample":
                stage = int(parts[1])
                idx = parts[3]
                rest = ".".join(parts[4:])
                put(f"downsample_layers.{stage}.{idx}.{rest}", v)
                continue

        # Stage blocks
        if k.startswith("stages.") and ".blocks." in k:
            parts = k.split(".")
            # stages.{stage}.blocks.{block}.{...}
            if len(parts) >= 5 and parts[2] == "blocks":
                stage = int(parts[1])
                block = int(parts[3])
                tail = parts[4:]
                # gamma
                if tail == ["gamma"]:
                    put(f"stages.{stage}.{block}.gamma", v)
                    continue
                # conv_dw.{weight,bias} -> dwconv.{weight,bias}
                if len(tail) == 2 and tail[0] == "conv_dw":
                    put(f"stages.{stage}.{block}.dwconv.{tail[1]}", v)
                    continue
                # norm.{weight,bias} -> norm.{weight,bias}
                if len(tail) == 2 and tail[0] == "norm":
                    put(f"stages.{stage}.{block}.norm.{tail[1]}", v)
                    continue
                # mlp.fc{1,2}.{weight,bias} -> pwconv{1,2}.{weight,bias}
                if len(tail) == 3 and tail[0] == "mlp" and tail[1] in {"fc1", "fc2"}:
                    fc = tail[1]
                    param = tail[2]
                    pw = "pwconv1" if fc == "fc1" else "pwconv2"
                    put(f"stages.{stage}.{block}.{pw}.{param}", v)
                    continue

    if not out:
        raise ValueError("No keys were mapped. Is this a timm ConvNeXt state_dict?")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, required=True, help="Output .pth with tensors-only state_dict")
    ap.add_argument("--model", type=str, default="convnext_tiny", help="timm model name")
    ap.add_argument("--in-chans", type=int, default=1, help="Input channels (depth=1)")
    ap.add_argument("--no-pretrained", action="store_true", help="Do not load timm pretrained weights")
    args = ap.parse_args()

    try:
        import timm  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("timm is required for this conversion script") from e

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    m = timm.create_model(
        str(args.model),
        pretrained=not bool(args.no_pretrained),
        in_chans=int(args.in_chans),
    )
    timm_state = {k: v.detach().cpu() for k, v in m.state_dict().items()}

    mapped = map_timm_convnext_state_dict_to_mgm_depth_backbone(timm_state)

    # Save tensors only so `torch.load(..., weights_only=True)` works.
    torch.save(mapped, out_path)

    print(f"[convert-timm-convnext->mgm-depth] model={args.model} pretrained={not args.no_pretrained} in_chans={args.in_chans}")
    print(f"[convert-timm-convnext->mgm-depth] timm_keys={len(timm_state)} mapped_keys={len(mapped)}")
    print(f"[convert-timm-convnext->mgm-depth] output={out_path}")


if __name__ == "__main__":
    main()
