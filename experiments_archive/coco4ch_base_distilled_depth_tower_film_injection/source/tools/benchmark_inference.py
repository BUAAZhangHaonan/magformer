#!/usr/bin/env python3
"""
MagFormer Inference Benchmark for Edge Deployment Optimization

Profiles per-component inference timing at specified resolution.
Uses torch.cuda.Event for precise GPU timing.

Usage:
    CUDA_VISIBLE_DEVICES=5 python tools/benchmark_inference.py \
        --config-file configs/v88_mbv3l_8dec_32k.yaml \
        --weights output/experiments/v88_mbv3l_8dec_32k_from_v60_dice20/checkpoint_iter_0031999.pth
"""

import argparse
import inspect
import os
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magformer.config import load_config
from magformer.models import build_model
from magformer.engine.utils import load_checkpoint


# ──────────────────────────────────────────────
# CUDA Event Timer
# ──────────────────────────────────────────────
class CUDATimer:
    """Precise GPU timing with CUDA events."""

    def __init__(self):
        self.events = OrderedDict()

    @contextmanager
    def measure(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self.events[name] = (start, end)

    def synchronize_and_report(self) -> Dict[str, float]:
        torch.cuda.synchronize()
        timings = {}
        for name, (start, end) in self.events.items():
            timings[name] = start.elapsed_time(end)
        return timings


# ──────────────────────────────────────────────
# Per-component profiling
# ──────────────────────────────────────────────
@torch.no_grad()
def profile_components(
    model: nn.Module,
    images: torch.Tensor,
    depths: torch.Tensor,
    warmup: int = 50,
    repeats: int = 100,
):
    """Profile each major component of MagFormer inference."""
    device = images.device

    # Warmup
    print(f"[Warmup] Running {warmup} iterations ...")
    for i in range(warmup):
        _ = model.forward_inference_exported(images, depths)
        if i % 10 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    print(f"[Warmup] Done.")

    # Timed runs - component level
    print(f"[Profile] Running {repeats} timed iterations (component-level) ...")
    all_runs = []

    for run_idx in range(repeats):
        timer = CUDATimer()

        # 1. Preprocessing (normalize)
        with timer.measure("0_preprocess"):
            images_norm = (images - model.pixel_mean) / model.pixel_std

        # 2. RGB Backbone (Swin-T)
        with timer.measure("1_rgb_backbone_swin_t"):
            rgb_features = model.rgb_backbone(images_norm)

        # 3. Depth Backbone (MBV3-Large)
        fusion_enabled = model.modality_fusion_enabled and model.depth_backbone_enabled

        if fusion_enabled:
            with timer.measure("2_depth_backbone_mbv3l"):
                depth_features = model.depth_backbone(depths)

            # 4. SA-Gate Fusion
            with timer.measure("3_fusion_sa_gate"):
                fused_features, confidence_maps, _ = model.fusion(
                    image_features=rgb_features,
                    depth_features=depth_features,
                    depth_raw=depths,
                    rgb_image=images_norm,
                    depth_noise_mask=None,
                )
        else:
            fused_features = rgb_features
            confidence_maps = None

        # 5. AGPE (if enabled)
        if getattr(model, 'agpe_enabled', False) and model.agpe is not None:
            with timer.measure("4_agpe"):
                _agpe_keys = ['res2', 'res3', 'res4', 'res5']
                _agpe_feats = [fused_features[k] for k in _agpe_keys if k in fused_features]
                if len(_agpe_feats) == model.agpe.num_levels:
                    _agpe_enhanced = model.agpe(_agpe_feats)
                    for _k, _f in zip(_agpe_keys, _agpe_enhanced):
                        fused_features[_k] = _f

        # 6. Pixel Decoder (MSDeformAttn)
        pd_kwargs = {
            "features": fused_features,
            "confidence_maps": confidence_maps,
            "depth_raw": depths,
            "padding_mask": None,
        }
        sig = inspect.signature(model.pixel_decoder.forward)
        if "depth_modulation_maps" in sig.parameters:
            pd_kwargs["depth_modulation_maps"] = confidence_maps

        with timer.measure("5_pixel_decoder_msdeform"):
            decoder_inputs = model.pixel_decoder(**pd_kwargs)

        pos_key_list = decoder_inputs.get("pos_key_list", None)

        # 7. Transformer Decoder (8 layers)
        with timer.measure("6_transformer_decoder"):
            outputs = model.decoder(
                memory=decoder_inputs["memory"],
                mask_features=decoder_inputs["mask_features"],
                multi_scale_features=decoder_inputs.get("multi_scale_features", None),
                multi_scale_pos=decoder_inputs.get("multi_scale_pos", None),
                pos_key=pos_key_list,
            )

        # 8. Post-processing (softmax, topk, mask threshold)
        with timer.measure("7_postprocess"):
            _ = model._inference_raw(outputs, images.shape)

        torch.cuda.synchronize()
        timings = timer.synchronize_and_report()
        all_runs.append(timings)

        if (run_idx + 1) % 20 == 0:
            print(f"  ... {run_idx + 1}/{repeats} done")

    # Aggregate
    component_names = list(all_runs[0].keys())
    avg_timings = {}
    std_timings = {}
    min_timings = {}
    max_timings = {}
    for name in component_names:
        vals = [run[name] for run in all_runs]
        avg_timings[name] = sum(vals) / len(vals)
        std_timings[name] = (sum((v - avg_timings[name])**2 for v in vals) / len(vals)) ** 0.5
        min_timings[name] = min(vals)
        max_timings[name] = max(vals)

    return avg_timings, std_timings, min_timings, max_timings


@torch.no_grad()
def measure_e2e_fps(model, images, depths, warmup=50, repeats=100):
    """Measure end-to-end FPS using forward_inference_exported."""
    print(f"[E2E] Measuring end-to-end FPS ({warmup} warmup + {repeats} timed) ...")

    for _ in range(warmup):
        _ = model.forward_inference_exported(images, depths)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = model.forward_inference_exported(images, depths)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_ms = sum(times) / len(times)
    std_ms = (sum((t - avg_ms)**2 for t in times) / len(times)) ** 0.5
    fps = 1000.0 / avg_ms
    return avg_ms, std_ms, fps, times


def print_report(
    avg_timings, std_timings, min_timings, max_timings,
    e2e_ms, e2e_std, e2e_fps,
    gpu_name, gpu_mem_mb, param_count,
    image_size, warmup, repeats,
):
    """Print formatted benchmark report."""
    total_component_ms = sum(avg_timings.values())

    print("\n")
    print("=" * 78)
    print("  MagFormer Inference Benchmark Report")
    print("=" * 78)
    print(f"  Resolution:   {image_size}x{image_size}")
    print(f"  GPU:          {gpu_name}")
    print(f"  Precision:    FP32")
    print(f"  Parameters:   {param_count / 1e6:.2f}M")
    print(f"  GPU Memory:   {gpu_mem_mb:.0f} MB (peak during inference)")
    print(f"  Warmup:       {warmup} iters")
    print(f"  Timed:        {repeats} iters")
    print()

    # ── End-to-End Summary ──
    print("  ─── Throughput Summary ───")
    print(f"  End-to-End Latency:  {e2e_ms:.1f} +/- {e2e_std:.1f} ms")
    print(f"  Throughput:          {e2e_fps:.2f} FPS")
    print(f"  Component Sum:       {total_component_ms:.1f} ms")
    print()

    # ── Per-component breakdown ──
    print(f"  {'Component':<35} {'Avg (ms)':>9} {'Std':>7} {'Min':>7} {'Max':>7} {'%Total':>8}")
    print("  " + "-" * 76)

    sorted_components = sorted(avg_timings.items(), key=lambda x: -x[1])

    for name, ms in sorted_components:
        pct = (ms / total_component_ms) * 100
        std = std_timings[name]
        mn = min_timings[name]
        mx = max_timings[name]
        bar = "#" * int(pct / 2)
        print(f"  {name:<35} {ms:>8.2f}  {std:>6.2f}  {mn:>6.2f}  {mx:>6.2f}  {pct:>6.1f}%  {bar}")

    print("  " + "-" * 76)
    print(f"  {'TOTAL (component sum)':<35} {total_component_ms:>8.2f}  {'':>7}  {'':>7}  {'':>7}  100.0%")
    print()

    # ── Grouped summary ──
    backbone_ms = avg_timings.get("1_rgb_backbone_swin_t", 0) + avg_timings.get("2_depth_backbone_mbv3l", 0)
    fusion_ms = avg_timings.get("3_fusion_sa_gate", 0)
    agpe_ms = avg_timings.get("4_agpe", 0)
    pixel_dec_ms = avg_timings.get("5_pixel_decoder_msdeform", 0)
    tf_dec_ms = avg_timings.get("6_transformer_decoder", 0)
    post_ms = avg_timings.get("7_postprocess", 0) + avg_timings.get("0_preprocess", 0)

    groups = [
        ("Backbones (Swin-T + MBV3-L)", backbone_ms),
        ("SA-Gate Fusion", fusion_ms),
        ("AGPE", agpe_ms),
        ("Pixel Decoder (MSDeformAttn)", pixel_dec_ms),
        ("Transformer Decoder (8L)", tf_dec_ms),
        ("Preprocess + Postprocess", post_ms),
    ]
    groups.sort(key=lambda x: -x[1])

    print("  ─── Grouped Breakdown ───")
    for gname, gms in groups:
        pct = (gms / total_component_ms) * 100
        print(f"  {gname:<40} {gms:>7.1f} ms  ({pct:>5.1f}%)")
    print()

    # ── Top 3 bottlenecks ──
    print("  ─── Top 3 Bottlenecks ───")
    for i, (name, ms) in enumerate(sorted_components[:3]):
        pct = (ms / total_component_ms) * 100
        print(f"    {i+1}. {name}: {ms:.1f} ms ({pct:.1f}%)")
    print()

    # ── Edge deployment assessment ──
    print("  ─── Edge Deployment Assessment ───")
    targets = [
        ("NVIDIA Jetson AGX Orin (~30W)", 30.0, 1000 / 30.0),
        ("NVIDIA Jetson Orin NX (~15W)", 60.0, 1000 / 60.0),
        ("NVIDIA Jetson Xavier NX (~15W)", 80.0, 1000 / 80.0),
        ("Intel Core i7 NUC (CPU only)", 200.0, 1000 / 200.0),
    ]
    for target_name, target_ms, target_fps in targets:
        ratio = e2e_ms / target_ms
        status = "PASS" if ratio <= 1.0 else f"NEED {ratio:.1f}x speedup"
        print(f"    {target_name:<40} {target_fps:>5.1f} FPS target -> {status}")

    print("=" * 78)
    print()


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description="Benchmark MagFormer inference speed")
    parser.add_argument("--config-file", required=True, help="Path to config yaml")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--image-size", type=int, default=1024, help="Input resolution")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=100, help="Timed iterations")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    gpu_total2 = torch.cuda.get_device_properties(0).total_memory / 1024**2
    gpu_total_mb = max(gpu_total, gpu_total2)
    print(f"\n[GPU] {gpu_name}  |  VRAM: {gpu_total_mb:.0f} MB")

    # Memory baseline
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated() / 1024**2

    # Load config and model
    print(f"[Config] Loading {args.config_file}")
    config = load_config(args.config_file)

    print("[Model] Building MagFormer ...")
    model = build_model(config)

    total_params, trainable_params = count_parameters(model)
    print(f"[Model] Parameters: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable")

    print(f"[Checkpoint] Loading {args.weights}")
    load_checkpoint(args.weights, model, strict=False)
    model = model.to(device).eval()

    mem_after_load = torch.cuda.memory_allocated() / 1024**2
    print(f"[Memory] After model load: {mem_after_load:.0f} MB")

    # Create dummy inputs
    H = W = args.image_size
    images = torch.randn(1, 3, H, W, device=device)
    depths = torch.randn(1, 1, H, W, device=device)

    # ── Component-level profiling ──
    torch.cuda.reset_peak_memory_stats()
    avg_timings, std_timings, min_timings, max_timings = profile_components(
        model, images, depths, warmup=args.warmup, repeats=args.repeats
    )
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    gpu_mem_used = peak_mem - mem_before

    # ── End-to-end FPS ──
    e2e_ms, e2e_std, e2e_fps, e2e_times = measure_e2e_fps(
        model, images, depths, warmup=args.warmup, repeats=args.repeats
    )

    # ── Print report ──
    print_report(
        avg_timings, std_timings, min_timings, max_timings,
        e2e_ms, e2e_std, e2e_fps,
        gpu_name, gpu_mem_used, total_params,
        args.image_size, args.warmup, args.repeats,
    )

    print("[Done] Benchmark complete.")


if __name__ == "__main__":
    main()
