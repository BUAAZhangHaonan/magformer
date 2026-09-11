#!/usr/bin/env python3
"""Build a TRT engine from the exported ONNX graph via the python API
(no torch-tensorrt), benchmark it, and verify predictions.

  python tools/trt/onnx_build_bench.py --onnx /tmp/c0_1024.onnx --precision fp32 --verify
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(SRC))

CFG = str(SRC / "configs" / "perf_c0_infer_bench.yaml")
WEIGHTS = "/home/hdd3/zhanghaonan/magformer/archive_20260906/staging_4028/home/g203-4028/magformer/output/experiments/next_stage/c0_corrected_300k_seed42/model_best.pth"
REF = Path("/home/hdd3/zhanghaonan/magformer/output/perf_20260911/ref_preds")


def build_engine(onnx_path, engine_path, fp16, workspace_gb=6):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)][:8]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errs))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    # strict fp32: ensure TF32 is off
    if hasattr(trt.BuilderFlag, "TF32"):
        cfg.clear_flag(trt.BuilderFlag.TF32)
    t0 = time.time()
    engine = builder.build_serialized_network(network, cfg)
    if engine is None:
        raise RuntimeError("engine build returned None")
    Path(engine_path).write_bytes(engine)
    print(f"[TRT] engine built in {time.time()-t0:.1f}s -> {engine_path} ({Path(engine_path).stat().st_size/1e6:.0f} MB)")


class TrtRunner:
    def __init__(self, engine_path):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def __call__(self, images_t, depths_t):
        import tensorrt as trt
        dev = torch.cuda.current_device()
        imgs = images_t if images_t.is_contiguous() else images_t.contiguous()
        dps = depths_t if depths_t.is_contiguous() else depths_t.contiguous()
        out = {}
        bindings = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                t = imgs if "images" in name else dps
                self.ctx.set_tensor_address(name, t.data_ptr())
                bindings.append((name, t))
            else:
                shape = tuple(self.ctx.get_tensor_shape(name))
                dt = torch.float16 if "16" in str(self.engine.get_tensor_dtype(name)) else torch.float32
                buf = torch.empty(shape, dtype=dt, device=imgs.device)
                self.ctx.set_tensor_address(name, buf.data_ptr())
                out[name] = buf
        ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT execute failed")
        self.stream.synchronize()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true", default=True)
    args = ap.parse_args()

    engine_path = str(SRC / "tools/trt" / (Path(args.onnx).stem + f"_{args.precision}.engine"))
    if not args.skip_build or not Path(engine_path).exists():
        build_engine(args.onnx, engine_path, fp16=args.precision == "fp16")

    dev = torch.device("cuda:0")
    runner = TrtRunner(engine_path)

    from magformer.config import load_config
    from magformer.models import build_model
    from magformer.engine.utils import load_checkpoint
    config = load_config(CFG)
    model = build_model(config)
    load_checkpoint(WEIGHTS, model, strict=False)
    model = model.to(dev).eval()

    half = args.precision == "fp16"
    dtype = torch.half if half else torch.float32
    im = torch.randn(1, 3, 1024, 1024, device=dev, dtype=dtype)
    dp = torch.rand(1, 1, 1024, 1024, device=dev, dtype=dtype)

    # bench decoder-outputs segment
    for _ in range(10):
        runner(im, dp)
    torch.cuda.synchronize()
    ts = []
    for _ in range(50):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        runner(im, dp)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    res = {"scope": "full", "precision": args.precision,
           "segment_latency_ms_p50": round(ts[25], 2),
           "segment_latency_ms_mean": round(sum(ts) / len(ts), 2),
           "fps": round(1000 / (sum(ts) / len(ts)), 2)}
    print(json.dumps(res, indent=1))
    (SRC / "tools/trt" / (Path(engine_path).name + ".bench.json")).write_text(json.dumps(res, indent=1))

    if args.verify:
        from magformer.data import CocoRgbdDataset
        from magformer.data.transforms import RGBDTransform
        from magformer.data.collate import collate_fn
        d = config.data
        transform = RGBDTransform(image_size=d.image_size, min_scale=d.min_scale, max_scale=d.max_scale,
                                  random_flip="none", rgb_brightness=0.0, rgb_contrast=0.0,
                                  rgb_saturation=0.0, rgb_hue=0.0, depth_scale=d.depth.scale,
                                  depth_shift=d.depth.shift, depth_clip_min=d.depth.clip_min,
                                  depth_clip_max=d.depth.clip_max, depth_norm=d.depth.norm,
                                  depth_per_sample_norm=False, is_train=False)
        ds = CocoRgbdDataset(dataset_root=d.dataset_root, ann_file="annotations/instances_val.validated.json",
                             split="val", transform=transform, is_train=False)
        max_score_diff, xor_total, ref_px, cat_mm = 0.0, 0, 0, 0
        for i in range(20):
            batch = collate_fn([ds[i]])
            im = batch["images"].to(dev); dp = batch["depths"].to(dev)
            outs = runner(im.to(dtype), dp.to(dtype))
            pl = outs[[k for k in outs if "logits" in k][0]].float()
            pm = outs[[k for k in outs if "masks" in k][0]].float()
            with torch.inference_mode():
                raw = model._inference_raw({"pred_logits": pl, "pred_masks": pm},
                                           tuple(im.shape), move_predictions_to_cpu=True)
            pred = raw["predictions"][0]
            ref = np.load(REF / f"pred_{i:04d}.npz")
            max_score_diff = max(max_score_diff, float(np.abs(pred["scores"].numpy() - ref["scores"]).max()))
            cat_mm += int((pred["category_ids"].numpy() != ref["category_ids"]).sum())
            m = pred["masks"].numpy().astype(np.uint8)
            xor_total += int((m != ref["masks"]).sum()); ref_px += int(ref["masks"].sum())
        vres = {"verify": "bit_identical" if (max_score_diff == 0 and xor_total == 0 and cat_mm == 0) else "differs",
                "scores_max_abs_diff": max_score_diff, "category_mismatches": cat_mm,
                "mask_xor_pixels": xor_total, "mask_diff_frac": xor_total / max(ref_px, 1)}
        print(json.dumps(vres, indent=1))
        (SRC / "tools/trt" / (Path(engine_path).name + ".verify.json")).write_text(json.dumps(vres, indent=1))


if __name__ == "__main__":
    main()
