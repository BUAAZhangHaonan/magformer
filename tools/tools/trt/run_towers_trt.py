#!/usr/bin/env python3
"""Final TensorRT assembly for c0 (v317): TRT-compiled RGB(Swin)+depth(MBv3L)
towers, eager fusion/decoder (full-graph TRT is numerically wrong: the
fusion/decoder segment contains data-dependent branches + op lowerings that
do not survive ONNX export; the towers are exact to conv-reassociation).

Benchmarks: towers segment eager vs TRT; full-model latency with the TRT
towers patched into _extract_dual_features; prediction verification vs the
eager fp32 reference.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "tools/trt"))

CFG = str(SRC / "configs" / "perf_c0_infer_bench.yaml")
WEIGHTS = "/home/hdd3/zhanghaonan/magformer/archive_20260906/staging_4028/home/g203-4028/magformer/output/experiments/next_stage/c0_corrected_300k_seed42/model_best.pth"
REF = Path("/home/hdd3/zhanghaonan/magformer/output/perf_20260911/ref_preds")
KEYS = ("res2", "res3", "res4", "res5")


class Towers(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.rgb = model.rgb_backbone
        self.depth = model.depth_backbone
        self.register_buffer("pm", model.pixel_mean.detach().clone())
        self.register_buffer("ps", model.pixel_std.detach().clone())

    def forward(self, images, depths):
        inorm = (images - self.pm.view(1, -1, 1, 1)) / self.ps.view(1, -1, 1, 1)
        r = self.rgb(inorm)
        d = self.depth(depths)
        return (*(r[k] for k in KEYS), *(d[k] for k in KEYS))


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return round(ts[len(ts) // 2], 2), round(sum(ts) / len(ts), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(SRC / "tools/trt" / "c0_towers_fp32.engine"))
    args = ap.parse_args()

    from onnx_build_bench import TrtRunner
    from magformer.config import load_config
    from magformer.models import build_model
    from magformer.engine.utils import load_checkpoint
    from magformer.data import CocoRgbdDataset
    from magformer.data.transforms import RGBDTransform
    from magformer.data.collate import collate_fn

    dev = torch.device("cuda:0")
    config = load_config(CFG)
    model = build_model(config)
    load_checkpoint(WEIGHTS, model, strict=False)
    model = model.to(dev).eval()

    d = config.data
    transform = RGBDTransform(image_size=d.image_size, min_scale=d.min_scale, max_scale=d.max_scale,
                              random_flip="none", rgb_brightness=0.0, rgb_contrast=0.0,
                              rgb_saturation=0.0, rgb_hue=0.0, depth_scale=d.depth.scale,
                              depth_shift=d.depth.shift, depth_clip_min=d.depth.clip_min,
                              depth_clip_max=d.depth.clip_max, depth_norm=d.depth.norm,
                              depth_per_sample_norm=False, is_train=False)
    ds = CocoRgbdDataset(dataset_root=d.dataset_root, ann_file="annotations/instances_val.validated.json",
                         split="val", transform=transform, is_train=False)
    batch = collate_fn([ds[0]])
    im = batch["images"].to(dev)
    dp = batch["depths"].to(dev)
    pad = batch.get("padding_masks")
    pad = pad.to(dev) if pad is not None else torch.zeros(1, 1024, 1024, dtype=torch.bool, device=dev)
    dvm = batch.get("depth_valid_masks")
    dvm = dvm.to(dev) if dvm is not None else torch.ones_like(dp, dtype=torch.bool)
    dnm = batch.get("noise_masks")
    dnm = dnm.to(dev) if dnm is not None else torch.zeros_like(dp, dtype=torch.bool)

    runner = TrtRunner(args.engine)
    towers = Towers(model).eval()

    # --- segment bench ---
    eager_p50, _ = bench(lambda: towers(im, dp))
    trt_p50, _ = bench(lambda: runner(im, dp))
    print(f"[TRT] towers segment: eager {eager_p50} ms -> TRT {trt_p50} ms ({eager_p50/trt_p50:.2f}x)")

    # --- full model, eager vs TRT-patched ---
    def full(m):
        with torch.inference_mode():
            return m.forward_inference_raw(im, dp, padding_masks=pad, depth_valid_masks=dvm,
                                           depth_noise_masks=dnm, move_predictions_to_cpu=False,
                                           gpu_export=True)
    full_eager_p50, _ = bench(lambda: full(model))

    orig = model._extract_dual_features

    def patched(images, depths, depth_valid_masks=None, depth_noise_masks=None):
        outs = runner(images, depths)
        rgb_features = {k: outs[f"rgb_{k}"] for k in KEYS}
        depth_features = {k: outs[f"dep_{k}"] for k in KEYS}
        images_norm = (images - model.pixel_mean.view(1, -1, 1, 1)) / model.pixel_std.view(1, -1, 1, 1)
        fused, conf, _l = model.fusion(
            image_features=rgb_features, depth_features=depth_features,
            depth_raw=depths, rgb_image=images_norm, depth_noise_mask=depth_noise_masks)
        return fused, conf, _l or {}

    model._extract_dual_features = patched
    full_trt_p50, _ = bench(lambda: full(model))
    print(f"[TRT] full model (gpu_export): eager {full_eager_p50} ms -> towers-TRT {full_trt_p50} ms ({full_eager_p50/full_trt_p50:.2f}x)")

    # --- prediction verification (20 ref images) ---
    max_score_diff, xor_total, ref_px, cat_mm = 0.0, 0, 0, 0
    for i in range(20):
        b = collate_fn([ds[i]])
        bi = b["images"].to(dev); bd = b["depths"].to(dev)
        bpad = b.get("padding_masks"); bpad = bpad.to(dev) if bpad is not None else torch.zeros(1, 1024, 1024, dtype=torch.bool, device=dev)
        bdvm = b.get("depth_valid_masks"); bdvm = bdvm.to(dev) if bdvm is not None else torch.ones_like(bd, dtype=torch.bool)
        bdnm = b.get("noise_masks"); bdnm = bdnm.to(dev) if bdnm is not None else torch.zeros_like(bd, dtype=torch.bool)
        with torch.inference_mode():
            raw = model.forward_inference_raw(bi, bd, padding_masks=bpad, depth_valid_masks=bdvm,
                                              depth_noise_masks=bdnm, move_predictions_to_cpu=True)
        pred = raw["predictions"][0]
        ref = np.load(REF / f"pred_{i:04d}.npz")
        max_score_diff = max(max_score_diff, float(np.abs(pred["scores"].numpy() - ref["scores"]).max()))
        cat_mm += int((pred["category_ids"].numpy() != ref["category_ids"]).sum())
        m = pred["masks"].numpy().astype(np.uint8)
        xor_total += int((m != ref["masks"]).sum()); ref_px += int(ref["masks"].sum())
    res = {
        "towers_segment_ms": {"eager": eager_p50, "trt": trt_p50, "speedup": round(eager_p50 / trt_p50, 2)},
        "full_model_ms": {"eager_gpu_export": full_eager_p50, "towers_trt": full_trt_p50,
                          "speedup": round(full_eager_p50 / full_trt_p50, 2)},
        "verify_scores_max_abs_diff": max_score_diff,
        "verify_mask_xor_frac": xor_total / max(ref_px, 1),
        "verify_category_mismatches": cat_mm,
    }
    print(json.dumps(res, indent=1))
    (SRC / "tools/trt" / "c0_towers_trt.results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
