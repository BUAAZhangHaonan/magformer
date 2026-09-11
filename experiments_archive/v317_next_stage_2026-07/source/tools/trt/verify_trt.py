#!/usr/bin/env python3
"""Verify a TRT engine's predictions against the eager fp32 reference.

Runs the TRT-compiled backbone->decoder graph on the SAME first-N val images
used to dump the reference predictions, applies the standard postprocess,
and compares per-image scores/category_ids/masks against the .npz refs.

  fp32-strict engine: expect bit-identical (scores exact, masks XOR == 0)
  fp16 engine: report diffs (accuracy gate decided by subset AP separately)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

CFG = str(SRC / "configs" / "perf_c0_infer_bench.yaml")
WEIGHTS = "/home/hdd3/zhanghaonan/magformer/archive_20260906/staging_4028/home/g203-4028/magformer/output/experiments/next_stage/c0_corrected_300k_seed42/model_best.pth"
REF = Path("/home/hdd3/zhanghaonan/magformer/output/perf_20260911/ref_preds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch_tensorrt
    from magformer.config import load_config
    from magformer.data import CocoRgbdDataset
    from magformer.data.transforms import RGBDTransform
    from magformer.data.collate import collate_fn
    from magformer.models import build_model
    from magformer.engine.utils import load_checkpoint

    config = load_config(CFG)
    device = torch.device("cuda:0")
    d = config.data
    transform = RGBDTransform(
        image_size=d.image_size, min_scale=d.min_scale, max_scale=d.max_scale,
        random_flip="none", rgb_brightness=0.0, rgb_contrast=0.0,
        rgb_saturation=0.0, rgb_hue=0.0, depth_scale=d.depth.scale,
        depth_shift=d.depth.shift, depth_clip_min=d.depth.clip_min,
        depth_clip_max=d.depth.clip_max, depth_norm=d.depth.norm,
        depth_per_sample_norm=False, is_train=False,
    )
    ds = CocoRgbdDataset(dataset_root=d.dataset_root, ann_file="annotations/instances_val.validated.json",
                         split="val", transform=transform, is_train=False)

    # eager model only for its postprocess (decoder weights are bypassed by TRT)
    model = build_model(config)
    load_checkpoint(WEIGHTS, model, strict=False)
    model = model.to(device).eval()

    trt_mod = torch_tensorrt.load(args.engine)

    max_score_diff = 0.0
    xor_total = 0
    ref_px_total = 0
    cat_mismatch = 0
    for i in range(args.n):
        batch = collate_fn([ds[i]])
        im = batch["images"].to(device)
        dp = batch["depths"].to(device)
        pad = batch.get("padding_masks")
        pad = pad.to(device) if pad is not None else torch.zeros(im.shape[0], *im.shape[-2:], dtype=torch.bool, device=device)
        dvm = batch.get("depth_valid_masks")
        dvm = dvm.to(device) if dvm is not None else torch.ones_like(dp, dtype=torch.bool)
        dnm = batch.get("noise_masks")
        dnm = dnm.to(device) if dnm is not None else torch.zeros_like(dp, dtype=torch.bool)
        with torch.inference_mode():
            pl, pm = trt_mod(im, dp)
            outputs = {"pred_logits": pl.float(), "pred_masks": pm.float()}
            raw = model._inference_raw(outputs, tuple(im.shape), move_predictions_to_cpu=True)
        pred = raw["predictions"][0]
        ref = np.load(REF / f"pred_{i:04d}.npz")
        max_score_diff = max(max_score_diff, float(np.abs(pred["scores"].numpy() - ref["scores"]).max()))
        cat_mismatch += int((pred["category_ids"].numpy() != ref["category_ids"]).sum())
        m = pred["masks"].numpy().astype(np.uint8)
        xor_total += int((m != ref["masks"]).sum())
        ref_px_total += int(ref["masks"].sum())

    result = {
        "engine": args.engine, "n_images": args.n,
        "scores_max_abs_diff": max_score_diff,
        "category_id_mismatches": cat_mismatch,
        "mask_xor_pixels": xor_total,
        "mask_diff_frac_of_ref": xor_total / max(ref_px_total, 1),
        "bit_identical": (max_score_diff == 0.0 and xor_total == 0 and cat_mismatch == 0),
    }
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
