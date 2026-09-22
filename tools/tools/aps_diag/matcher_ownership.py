#!/usr/bin/env python3
"""Matcher-ownership diagnostic for AP_s (mechanism-1 direct evidence).

For N val images: run the TRAINING-format forward (decoder raw outputs,
dropout disabled), run the Hungarian matcher, and measure per GT instance:
  - matched-query mask IoU (what the matcher actually chose)
  - best-query mask IoU (max over all 200 queries — capacity that exists)
  - matched-query class score
aggregated by GT size bucket. The gap (best - matched) is assignment loss;
(best < 0.5) is decoder/feature capacity shortfall.

IoU computed at 512x512 (pred: 256-grid sigmoid bilinear-up 2x > 0.5;
GT: 1024 INTER_AREA down > 0.5).

Usage:
  CUDA_VISIBLE_DEVICES=6 python matcher_ownership.py [--num-images 300]
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from pycocotools import mask as maskUtils

SOURCE = "/home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/source"
sys.path.insert(0, SOURCE)

from magformer.config import load_config, set_seed  # noqa: E402
from magformer.data import CocoRgbdDataset  # noqa: E402
from magformer.data.transforms import RGBDTransform  # noqa: E402
from magformer.data.collate import collate_fn  # noqa: E402
from magformer.models import build_model  # noqa: E402
from magformer.engine.utils import load_checkpoint  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

CONFIG = f"{SOURCE}/configs/perf_c0_infer_bench.yaml"
CKPT = (
    "/home/hdd3/zhanghaonan/magformer/archive_20260906/staging_4028/home/"
    "g203-4028/magformer/output/experiments/next_stage/c0_corrected_300k_seed42/model_best.pth"
)
GT = (
    "/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/"
    "annotations/instances_val.validated.json"
)


def bucket(area):
    if area < 64:
        return "xtiny<64"
    if area < 256:
        return "tiny64-256"
    if area < 1024:
        return "small256-1024"
    if area < 9216:
        return "medium"
    return "large"


def iou_matrix(pred_masks_512, gt_masks_512):
    """pred (Q,512,512) float/bool, gt (K,512,512) bool -> (Q,K) IoU."""
    p = pred_masks_512.flatten(1).float()
    g = gt_masks_512.flatten(1).float()
    inter = p @ g.T
    pa = p.sum(1, keepdim=True)
    ga = g.sum(1, keepdim=True).T
    union = pa + ga - inter
    return torch.where(union > 0, inter / union.clamp(min=1), torch.zeros_like(inter))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=300)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--out", default="/home/hdd3/zhanghaonan/magformer/output/aps_20260913/matcher_ownership/results.json")
    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda:0")

    config = load_config(args.config, overrides={"runtime": {"gpus": [0]}})
    model = build_model(config)
    load_checkpoint(args.checkpoint, model, strict=False)
    model = model.to(device)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()

    data_cfg = config.data
    dataset = CocoRgbdDataset(
        dataset_root=data_cfg.dataset_root,
        ann_file=data_cfg.val_ann,
        split=data_cfg.val_split,
        transform=None,
        is_train=False,
    )
    dataset.transform = RGBDTransform(
        image_size=data_cfg.image_size, min_scale=data_cfg.min_scale,
        max_scale=data_cfg.max_scale, random_flip="none",
        rgb_brightness=0.0, rgb_contrast=0.0, rgb_saturation=0.0, rgb_hue=0.0,
        depth_scale=data_cfg.depth.scale, depth_shift=data_cfg.depth.shift,
        depth_clip_min=data_cfg.depth.clip_min, depth_clip_max=data_cfg.depth.clip_max,
        depth_norm=data_cfg.depth.norm,
        depth_per_sample_norm=getattr(data_cfg.depth, "per_sample_norm", True),
        is_train=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                        collate_fn=collate_fn, pin_memory=True)

    gt_by_img = defaultdict(list)
    with open(GT) as f:
        for a in json.load(f)["annotations"]:
            gt_by_img[a["image_id"]].append(a)

    rec = defaultdict(list)  # bucket -> list of (matched_iou, best_iou, cls)
    n_done = 0
    for batch in loader:
        if n_done >= args.num_images:
            break
        img_id = int(batch["image_ids"][0])
        anns = gt_by_img.get(img_id, [])
        if not anns:
            continue
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        with torch.no_grad():
            fused, conf, _ = model._extract_dual_features(images, depths, None, None)
            if getattr(model, "agpe_enabled", False) and model.agpe is not None:
                keys = ["res2", "res3", "res4", "res5"]
                feats = [fused[k] for k in keys if k in fused]
                if len(feats) == model.agpe.num_levels:
                    enh = model.agpe(feats)
                    for k, e in zip(keys, enh):
                        fused[k] = e
            dec_in = model.pixel_decoder(**model._pixel_decoder_forward_kwargs(
                model.pixel_decoder, features=fused, confidence_maps=conf,
                depth_modulation_maps=conf, depth_raw=depths, padding_mask=None))
            outputs = model.decoder(
                memory=dec_in["memory"], mask_features=dec_in["mask_features"],
                multi_scale_features=dec_in.get("multi_scale_features"),
                multi_scale_pos=dec_in.get("multi_scale_pos"),
                multi_scale_padding_masks=dec_in["multi_scale_padding_masks"],
                pos_key=dec_in.get("pos_key_list"), depth_raw=depths)

            # GT masks at 1024
            h = w = 1024
            gt_full = np.stack([
                maskUtils.decode(maskUtils.merge(maskUtils.frPyObjects(a["segmentation"], h, w)))
                for a in anns
            ])
            gt_t = torch.from_numpy(gt_full).to(device).bool()  # (K,1024,1024)
            targets = [{
                "labels": torch.zeros(len(anns), dtype=torch.long, device=device),
                "masks": gt_t,
            }]
            indices = model.criterion.matcher(outputs, targets)
            q_idx, g_idx = indices[0]

            # pred masks (Q,256,256) -> 512 binary
            pm = outputs["pred_masks"][0].sigmoid()  # (Q,256,256)
            pm = torch.nn.functional.interpolate(
                pm[None], size=(512, 512), mode="bilinear", align_corners=False)[0] > 0.5
            gt512 = torch.nn.functional.interpolate(
                gt_t[None].float(), size=(512, 512), mode="area")[0] > 0.5
            ious = iou_matrix(pm, gt512)  # (Q,K)
            cls = outputs["pred_logits"][0].sigmoid()[:, 0]  # (Q,) objectness

            q2g = {int(g): int(q) for q, g in zip(q_idx.tolist(), g_idx.tolist())}
            for k, a in enumerate(anns):
                best = float(ious[:, k].max())
                matched = float(ious[q2g[k], k]) if k in q2g else 0.0
                c = float(cls[q2g[k]]) if k in q2g else 0.0
                rec[bucket(a["area"])].append((matched, best, c))
        n_done += 1
        if n_done % 50 == 0:
            print(f"{n_done}/{args.num_images}", flush=True)

    report = {}
    print(f"\n{'bucket':<14} {'n':>6} {'match>=.5':>9} {'best>=.5':>9} {'mean_matched':>12} {'mean_best':>10} {'gap':>7} {'mean_cls':>9}")
    for b, rows in sorted(rec.items()):
        m = np.array([r[0] for r in rows]); bb = np.array([r[1] for r in rows]); cc = np.array([r[2] for r in rows])
        row = {
            "n": len(rows),
            "frac_matched_iou_ge_0.5": float((m >= 0.5).mean()),
            "frac_best_iou_ge_0.5": float((bb >= 0.5).mean()),
            "mean_matched_iou": float(m.mean()),
            "mean_best_iou": float(bb.mean()),
            "assignment_gap": float((bb - m).mean()),
            "mean_matched_cls": float(cc.mean()),
        }
        report[b] = row
        print(f"{b:<14} {len(rows):>6} {row['frac_matched_iou_ge_0.5']:>9.3f} {row['frac_best_iou_ge_0.5']:>9.3f} {row['mean_matched_iou']:>12.3f} {row['mean_best_iou']:>10.3f} {row['assignment_gap']:>7.3f} {row['mean_matched_cls']:>9.3f}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
