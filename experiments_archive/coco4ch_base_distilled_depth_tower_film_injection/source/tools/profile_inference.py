#!/usr/bin/env python3
"""
MAGFormer Inference Profiler

Profiles per-component inference timing at 1024x1024 resolution.
Uses torch.cuda.Event for precise GPU timing and torch.profiler
for detailed operator-level breakdown.

Usage:
    CUDA_VISIBLE_DEVICES=4 python tools/profile_inference.py \
        --config-file configs/finetune_1k_full_1024_v15.yaml \
        --weights output/experiments/20260513_1k_finetune_full_1024_v14/model_best.pth
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, List, Any, Optional

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
            ms = start.elapsed_time(end)
            timings[name] = ms
        return timings


# ──────────────────────────────────────────────
# Per-component profiling via manual hooks
# ──────────────────────────────────────────────
@torch.no_grad()
def profile_components(model: nn.Module, images: torch.Tensor, depths: torch.Tensor,
                       warmup: int = 3, repeats: int = 10):
    """Profile each major component of MagFormer inference."""
    device = images.device

    # Warmup
    print(f"[Warmup] Running {warmup} iterations ...")
    for _ in range(warmup):
        _ = model.forward_inference_decoder_outputs(images, depths)
        torch.cuda.synchronize()

    # Timed runs
    print(f"[Profile] Running {repeats} timed iterations ...")
    all_runs = []

    for run_idx in range(repeats):
        timer = CUDATimer()

        # 1. RGB Backbone
        with timer.measure("1_rgb_backbone"):
            images_norm = (images - model.pixel_mean) / model.pixel_std
            rgb_features = model.rgb_backbone(images_norm)

        # 2. Depth Backbone
        fusion_enabled = bool(getattr(model, "modality_fusion_enabled", True)) and bool(
            getattr(model, "depth_backbone_enabled", True)
        )

        if fusion_enabled:
            with timer.measure("2_depth_backbone"):
                depth_features = model.depth_backbone(depths)

            # 3. SA-Gate Fusion
            with timer.measure("3_sa_gate_fusion"):
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

        # 3b. AGPE (if enabled)
        if getattr(model, 'agpe_enabled', False) and model.agpe is not None:
            with timer.measure("3b_agpe"):
                _agpe_keys = ['res2', 'res3', 'res4', 'res5']
                _agpe_feats = [fused_features[k] for k in _agpe_keys if k in fused_features]
                if len(_agpe_feats) == model.agpe.num_levels:
                    _agpe_enhanced = model.agpe(_agpe_feats)
                    for _k, _f in zip(_agpe_keys, _agpe_enhanced):
                        fused_features[_k] = _f

        # 4. Pixel Decoder
        import inspect
        pd_kwargs = {
            "features": fused_features,
            "confidence_maps": confidence_maps,
            "depth_raw": depths,
            "padding_mask": None,
        }
        sig = inspect.signature(model.pixel_decoder.forward)
        if "depth_modulation_maps" in sig.parameters:
            pd_kwargs["depth_modulation_maps"] = confidence_maps

        with timer.measure("4_pixel_decoder"):
            decoder_inputs = model.pixel_decoder(**pd_kwargs)

        pos_key_list = decoder_inputs.get("pos_key_list", None)

        # 5. Transformer Decoder
        with timer.measure("5_transformer_decoder"):
            outputs = model.decoder(
                memory=decoder_inputs["memory"],
                mask_features=decoder_inputs["mask_features"],
                multi_scale_features=decoder_inputs.get("multi_scale_features", None),
                multi_scale_pos=decoder_inputs.get("multi_scale_pos", None),
                pos_key=pos_key_list,
            )

        # 6. Post-processing (mask generation + scoring)
        with timer.measure("6_postprocess"):
            _ = model._inference_raw(outputs, images.shape)

        # Total
        torch.cuda.synchronize()
        timings = timer.synchronize_and_report()
        all_runs.append(timings)

    # Aggregate
    component_names = list(all_runs[0].keys())
    avg_timings = {}
    std_timings = {}
    for name in component_names:
        vals = [run[name] for run in all_runs]
        avg_timings[name] = sum(vals) / len(vals)
        std_timings[name] = (sum((v - avg_timings[name])**2 for v in vals) / len(vals)) ** 0.5

    return avg_timings, std_timings


def print_report(avg_timings: Dict[str, float], std_timings: Dict[str, float],
                 total_fps: float, avg_total_ms: float, gpu_mem_mb: float,
                 image_size: int):
    """Print formatted profiling report."""
    print("\n" + "=" * 70)
    print(f"  MagFormer Inference Profile  |  {image_size}x{image_size}  |  FP32")
    print("=" * 70)

    # Total
    print(f"\n  Total Inference Time:  {avg_total_ms:.1f} ms")
    print(f"  Throughput:            {total_fps:.2f} FPS")
    print(f"  GPU Memory Used:       {gpu_mem_mb:.0f} MB")
    print(f"  Target:                1.00 FPS (1000 ms)")
    print(f"  Gap to Target:         {max(0, 1000.0 - avg_total_ms):.1f} ms {'OK' if avg_total_ms <= 1000 else 'SLOW'}")

    # Per-component breakdown
    print(f"\n  {'Component':<30} {'Time (ms)':>10} {'Std (ms)':>10} {'% Total':>10}")
    print("  " + "-" * 64)

    total_component_ms = sum(avg_timings.values())
    sorted_components = sorted(avg_timings.items(), key=lambda x: -x[1])

    for name, ms in sorted_components:
        pct = (ms / total_component_ms) * 100 if total_component_ms > 0 else 0
        std = std_timings.get(name, 0)
        bar = "#" * int(pct / 2)
        print(f"  {name:<30} {ms:>8.1f}   {std:>8.1f}   {pct:>6.1f}%  {bar}")

    print("  " + "-" * 64)
    print(f"  {'TOTAL':<30} {total_component_ms:>8.1f}   {'':>8}   100.0%")

    # Top 3 bottlenecks
    print(f"\n  Top 3 Bottlenecks:")
    for i, (name, ms) in enumerate(sorted_components[:3]):
        pct = (ms / total_component_ms) * 100
        print(f"    {i+1}. {name}: {ms:.1f} ms ({pct:.1f}%)")

    print("=" * 70 + "\n")


# ──────────────────────────────────────────────
# torch.profiler detailed operator profile
# ──────────────────────────────────────────────
@torch.no_grad()
def profile_operators(model: nn.Module, images: torch.Tensor, depths: torch.Tensor):
    """Run torch.profiler for detailed operator-level profile."""
    print("[Profiler] Running torch.profiler (1 warmup + 1 profiled) ...")

    # Warmup
    _ = model.forward_inference_decoder_outputs(images, depths)
    torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        _ = model.forward_inference_decoder_outputs(images, depths)

    # Print top operators by CUDA time
    print("\n  Top 20 operators by CUDA time:")
    print("  " + "-" * 90)
    key_averages = prof.key_averages()
    key_averages.sort(key=lambda x: x.cuda_time_total if hasattr(x, "cuda_time_total") else 0, reverse=True)

    total_cuda_ms = sum(ka.cuda_time_total if hasattr(ka, "cuda_time_total") else 0 for ka in key_averages) / 1000.0

    for i, ka in enumerate(key_averages[:20]):
        cuda_ms = ka.cuda_time_total if hasattr(ka, "cuda_time_total") else 0 / 1000.0
        pct = (cuda_ms / total_cuda_ms * 100) if total_cuda_ms > 0 else 0
        cpu_ms = ka.cpu_time_total / 1000.0
        print(f"  {i+1:>3}. {ka.key:<55} CUDA {cuda_ms:>7.1f}ms ({pct:>5.1f}%)  CPU {cpu_ms:>7.1f}ms")

    print("  " + "-" * 90)
    print(f"  Total CUDA time: {total_cuda_ms:.1f} ms\n")

    return prof


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Profile MagFormer inference")
    parser.add_argument("--config-file", default="configs/finetune_1k_full_1024_v14.yaml")
    parser.add_argument("--weights",
                        default="output/experiments/20260513_1k_finetune_full_1024_v14/model_best.pth")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--profile-operators", action="store_true", default=True,
                        help="Run torch.profiler for operator-level breakdown")
    parser.add_argument("--no-profile-operators", dest="profile_operators", action="store_false")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"\n[GPU] {gpu_name}  |  Total: {gpu_total:.0f} MB")

    # Record baseline memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated() / 1024**2

    # Load config and model
    print(f"[Config] Loading {args.config_file}")
    config = load_config(args.config_file)

    print(f"[Model] Building MagFormer ...")
    model = build_model(config)

    print(f"[Checkpoint] Loading {args.weights}")
    load_checkpoint(args.weights, model, strict=False)
    model = model.to(device).eval()

    mem_after_load = torch.cuda.memory_allocated() / 1024**2
    print(f"[Memory] Model loaded: {mem_after_load:.0f} MB")

    # Create dummy inputs
    H = W = args.image_size
    images = torch.randn(1, 3, H, W, device=device)
    depths = torch.randn(1, 1, H, W, device=device)

    # ── Component-level profiling ──
    print(f"\n{'='*70}")
    print(f"  Phase 1: Per-Component Timing ({args.repeats} iterations)")
    print(f"{'='*70}")

    torch.cuda.reset_peak_memory_stats()
    avg_timings, std_timings = profile_components(
        model, images, depths, warmup=args.warmup, repeats=args.repeats
    )

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    total_component_ms = sum(avg_timings.values())

    # Also measure end-to-end with full forward_inference_exported
    print(f"\n[End-to-End] Measuring full inference pipeline ...")
    # Warmup
    for _ in range(args.warmup):
        _ = model.forward_inference_exported(images, depths)
        torch.cuda.synchronize()

    e2e_times = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = model.forward_inference_exported(images, depths)
        end.record()
        torch.cuda.synchronize()
        e2e_times.append(start.elapsed_time(end))

    avg_e2e_ms = sum(e2e_times) / len(e2e_times)
    fps = 1000.0 / avg_e2e_ms

    gpu_mem_used = peak_mem - mem_before

    print_report(avg_timings, std_timings, fps, avg_e2e_ms, gpu_mem_used, args.image_size)

    # ── Operator-level profiling ──
    if args.profile_operators:
        print(f"\n{'='*70}")
        print(f"  Phase 2: Operator-Level Profiling (torch.profiler)")
        print(f"{'='*70}")
        prof = profile_operators(model, images, depths)

    print("\n[Done] Profiling complete.")


if __name__ == "__main__":
    main()
