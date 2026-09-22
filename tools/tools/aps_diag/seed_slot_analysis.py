#!/usr/bin/env python3
"""Which query slots own small GT in r2b — do seeded slots (136-199) cover
them, and what cls scores do they carry? Mechanism readout for the R2 null."""
import argparse, json, sys
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

SOURCE = "/home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/source"
sys.path.insert(0, SOURCE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-images", type=int, default=150)
    ap.add_argument("--seed-start", type=int, default=136)
    args = ap.parse_args()

    from magformer.config import load_config, set_seed
    from magformer.data import CocoRgbdDataset
    from magformer.data.transforms import RGBDTransform
    from magformer.data.collate import collate_fn
    from magformer.models import build_model
    from magformer.engine.utils import load_checkpoint
    from torch.utils.data import DataLoader
    from pycocotools import mask as maskUtils

    set_seed(42)
    cfg = load_config(args.config, overrides={"runtime": {"gpus": [0]}})
    m = build_model(cfg)
    ck = load_checkpoint(args.checkpoint, m, strict=False)
    # raw weights (not EMA): train.py checkpoints store model_state_dict raw
    m = m.cuda().eval()

    dc = cfg.data
    ds = CocoRgbdDataset(dataset_root=dc.dataset_root, ann_file=dc.val_ann, split=dc.val_split, transform=None, is_train=False)
    ds.transform = RGBDTransform(image_size=dc.image_size, min_scale=dc.min_scale, max_scale=dc.max_scale, random_flip="none",
        rgb_brightness=0.0, rgb_contrast=0.0, rgb_saturation=0.0, rgb_hue=0.0, depth_scale=dc.depth.scale, depth_shift=dc.depth.shift,
        depth_clip_min=dc.depth.clip_min, depth_clip_max=dc.depth.clip_max, depth_norm=dc.depth.norm,
        depth_per_sample_norm=getattr(dc.depth, "per_sample_norm", True), is_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    gt_by_img = defaultdict(list)
    with open("/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/annotations/instances_val.validated.json") as f:
        for a in json.load(f)["annotations"]:
            gt_by_img[a["image_id"]].append(a)

    stats = {"learned": [0,0,0.0,0.0], "seeded": [0,0,0.0,0.0]}  # n, covered@0.5, sum_best_iou, sum_cls
    probe_hits = [0,0]  # small GT with a probe peak within bbox, n
    n=0
    for batch in loader:
        if n >= args.num_images: break
        img_id = int(batch["image_ids"][0])
        anns = [a for a in gt_by_img.get(img_id, []) if a["area"] < 1024]
        if not anns: continue
        images = batch["images"].cuda(); depths = batch["depths"].cuda()
        with torch.inference_mode():
            fused, conf, _ = m._extract_dual_features(images, depths, None, None)
            keys=["res2","res3","res4","res5"]; enh=m.agpe([fused[k] for k in keys])
            for k,e in zip(keys,enh): fused[k]=e
            dec_in = m.pixel_decoder(**m._pixel_decoder_forward_kwargs(m.pixel_decoder, features=fused, confidence_maps=conf,
                depth_modulation_maps=conf, depth_raw=depths, padding_mask=None))
            probe_logits, _ = m._probe_outputs(dec_in, training=False)
            out = m.decoder(memory=dec_in["memory"], mask_features=dec_in["mask_features"],
                multi_scale_features=dec_in.get("multi_scale_features"), multi_scale_pos=dec_in.get("multi_scale_pos"),
                multi_scale_padding_masks=dec_in["multi_scale_padding_masks"], pos_key=dec_in.get("pos_key_list"),
                depth_raw=depths, mask_features_hi=dec_in.get("mask_features_hi"),
                probe_obj=probe_logits[:,0], probe_fg=probe_logits[:,1].sigmoid(), seed_active=True)
        pm = out["pred_masks"][0].float().sigmoid()
        pm = torch.nn.functional.interpolate(pm[None], size=(512,512), mode="bilinear", align_corners=False)[0] > 0.5
        cls = out["pred_logits"][0].float().sigmoid()[:,0]
        # probe peaks
        p = probe_logits[0,0].sigmoid()
        pk = torch.nn.functional.max_pool2d(p[None, None], 3, 1, 1)[0, 0]
        peaks = ((p == pk) & (p > 0.25)).nonzero().float()  # (K,2) yx at 256
        for a in anns:
            g = maskUtils.decode(maskUtils.merge(maskUtils.frPyObjects(a["segmentation"],1024,1024)))
            g512 = torch.from_numpy(cv2_style_down(g)).cuda().bool()
            ious = mask_iou(pm, g512)  # (200,)
            best_i, best_q = ious.max(0)
            grp = "seeded" if int(best_q) >= args.seed_start else "learned"
            s = stats[grp]; s[0]+=1; s[2]+=float(best_i); s[3]+=float(cls[best_q])
            if float(best_i) >= 0.5: s[1]+=1
            # probe peak inside bbox (256-grid coords)
            x0,y0,w,h = a["bbox"]
            if len(peaks) > 0:
                inside = ((peaks[:,1] >= x0/4) & (peaks[:,1] <= (x0+w)/4) & (peaks[:,0] >= y0/4) & (peaks[:,0] <= (y0+h)/4)).any() if len(peaks) else False
                probe_hits[0] += int(bool(inside))
            probe_hits[1] += 1
        n+=1
        if n % 50 == 0: print(f"{n} imgs", flush=True)
    for grp,(n_,cov,si,sc) in stats.items():
        if n_:
            print(f"{grp:>8}: n={n_} bestIoU>=0.5={cov/n_:.3f} mean_best_iou={si/n_:.3f} mean_cls_of_best={sc/n_:.3f}")
    print(f"probe peak inside small-GT bbox: {probe_hits[0]}/{probe_hits[1]} = {probe_hits[0]/max(probe_hits[1],1):.3f}")

def cv2_style_down(g):
    import cv2, numpy as np
    return cv2.resize(g.astype("float32"), (512,512), interpolation=cv2.INTER_AREA)

def mask_iou(pm, g):
    p = pm.flatten(1).float(); q = g.flatten().float()
    inter = p @ q; union = p.sum(1) + q.sum() - inter
    return torch.where(union>0, inter/union.clamp(min=1), torch.zeros_like(inter))

if __name__ == "__main__":
    main()
