#!/usr/bin/env python3
"""Build + bench a TensorRT engine for the c0 (v317) MAGFormer inference graph.

Requires: tensorrt 10.x + torch-tensorrt 2.5.x (see repo docs for the g203
proxy install recipe). Compiles images/depths -> (pred_logits, pred_masks)
with automatic partitioning: unsupported segments (e.g. the MSDeformAttn
custom CUDA op) stay in torch, everything else becomes TRT engines.

Modes:
  --precision fp32   strict fp32 (use_tf32=False) -> outputs bit-comparable
  --precision fp16   TRT fp16 (accuracy gate required)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

CFG = str(SRC / "configs" / "perf_c0_infer_bench.yaml")
WEIGHTS = "/home/hdd3/zhanghaonan/magformer/archive_20260906/staging_4028/home/g203-4028/magformer/output/experiments/next_stage/c0_corrected_300k_seed42/model_best.pth"


class BackboneToDecoder(torch.nn.Module):
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
    ap.add_argument("--engine", default=None)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--ir", choices=["ts", "dynamo"], default="ts")
    args = ap.parse_args()
    if args.engine is None:
        args.engine = str(SRC / "tools/trt" / f"c0_{args.image_size}_{args.precision}.ts")

    import torch_tensorrt
    from magformer.config import load_config
    from magformer.models import build_model
    from magformer.engine.utils import load_checkpoint

    config = load_config(CFG)
    model = build_model(config)
    load_checkpoint(WEIGHTS, model, strict=False)
    model = model.to("cuda").eval()
    wrap = BackboneToDecoder(model).eval()

    S = args.image_size
    half = args.precision == "fp16"
    dtype = torch.half if half else torch.float32
    inputs = [
        torch_tensorrt.Input((1, 3, S, S), dtype=dtype),
        torch_tensorrt.Input((1, 1, S, S), dtype=dtype),
    ]
    t0 = time.time()
    if args.ir == "ts":
        # TS frontend: jit-trace then partition; friendly to numpy/python in forwards
        example_im = torch.randn(1, 3, S, S, device="cuda")
        example_dp = torch.rand(1, 1, S, S, device="cuda")
        with torch.no_grad():
            traced = torch.jit.trace(wrap, (example_im, example_dp), check_trace=False)
        traced = torch.jit.freeze(traced.eval().cuda())
        trt_mod = torch_tensorrt.compile(
            traced,
            inputs=inputs,
            enabled_precisions={torch_tensorrt.dtype.half} if half else {torch_tensorrt.dtype.fp32},
            workspace_size=4 << 30,
            min_block_size=2,
            disable_tf32=True,
            ir="ts",
        )
    else:
        trt_mod = torch_tensorrt.compile(
            wrap,
            inputs=inputs,
            enabled_precisions={torch_tensorrt.dtype.half} if half else {torch_tensorrt.dtype.fp32},
            workspace_size=4 << 30,
            min_block_size=2,
            use_tf32=False,
            require_full_compilation=False,
            cache_built_engines=False,
        )
    compile_s = time.time() - t0
    print(f"[TRT] compiled ({args.precision}) in {compile_s:.1f}s")
    torch_tensorrt.save(trt_mod, args.engine)
    print("[TRT] saved", args.engine)

    if args.bench:
        im = torch.randn(1, 3, S, S, device="cuda", dtype=dtype)
        dp = torch.rand(1, 1, S, S, device="cuda", dtype=dtype)
        with torch.inference_mode():
            for _ in range(10):
                trt_mod(im, dp)
            torch.cuda.synchronize()
            ts = []
            for _ in range(50):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                trt_mod(im, dp)
                torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        res = {"precision": args.precision, "latency_ms_p50": round(ts[25], 2),
               "latency_ms_mean": round(sum(ts) / len(ts), 2),
               "fps": round(1000 / (sum(ts) / len(ts)), 2),
               "compile_s": round(compile_s, 1)}
        print(json.dumps(res, indent=1))
        Path(str(args.engine) + ".bench.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
