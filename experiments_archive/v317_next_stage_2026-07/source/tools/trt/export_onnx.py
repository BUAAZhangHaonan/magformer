#!/usr/bin/env python3
"""ONNX export for the MAGFormer c0 (v317) inference graph.

Exports the network up to (pred_logits, pred_masks@low-res) — the batched
post-processing (top-k, upsample, sigmoid) stays in torch on top of the
returned tensors, so no custom NMS plugin is needed.

NOTE (2026-09-11): TensorRT libraries are not installable on this machine
(NVIDIA domains blocked). This export + tools/trt/build_trt.py are provided so
that on a network-unrestricted host the TRT engine is one command away:
    python trt/export_onnx.py --onnx c0_1024.onnx
    python trt/build_trt.py --onnx c0_1024.onnx --engine c0_1024_fp16.engine
    python trt/bench_trt.py  --engine c0_1024_fp16.engine
"""
import argparse
import sys
from pathlib import Path

SRC = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

import torch

CFG = str(SRC / "configs" / "perf_c0_infer_bench.yaml")
WEIGHTS = "/home/hdd3/zhanghaonan/magformer/archive_20260906/staging_4028/home/g203-4028/magformer/output/experiments/next_stage/c0_corrected_300k_seed42/model_best.pth"


class BackboneToDecoder(torch.nn.Module):
    """images/depths -> (pred_logits, pred_masks_lowres). Postprocess-free."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, depths):
        outputs = self.model.forward_inference_decoder_outputs(
            images=images, depths=depths
        )
        return outputs["pred_logits"], outputs["pred_masks"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="c0_1024.onnx")
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    from magformer.config import load_config
    from magformer.models import build_model
    from magformer.engine.utils import load_checkpoint

    config = load_config(CFG)
    model = build_model(config)
    load_checkpoint(WEIGHTS, model, strict=False)
    model = model.to("cuda").eval()

    # NOTE: MSDeformAttn runs through a custom CUDA op; the ONNX graph will
    # contain it as a custom op "MultiScaleDeformableAttention::ms_deform_attn"
    # (or fall back per build_trt.py's partitioning). build_trt.py keeps that
    # layer in torch via torch-tensorrt partitioning when the plugin is absent.
    wrap = BackboneToDecoder(model).eval()
    im = torch.randn(1, 3, args.image_size, args.image_size, device="cuda")
    dp = torch.rand(1, 1, args.image_size, args.image_size, device="cuda")

    with torch.inference_mode():
        torch.onnx.export(
            wrap, (im, dp), args.onnx,
            input_names=["images", "depths"],
            output_names=["pred_logits", "pred_masks"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,  # fixed 1024 for max performance
        )
    print("exported", args.onnx)


if __name__ == "__main__":
    main()
