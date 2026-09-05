"""Pretrain the SGF DepthTower on a mask-distillation proxy.

Task: from the z-scored depth channel alone, predict the union instance mask
(deep supervision at the 4 tower scales, BCE + soft dice). This gives the
side-path useful features from step 0 instead of random init — the diagnosed
failure mode of every injection arm so far.

Usage: CUDA_VISIBLE_DEVICES=0 python pretrain_tower.py [--iters 8000]
Output: /home/hdd1/wanghaoran/magformer/pretrained/depth_tower_distilled.pth
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SRC = "/home/hdd1/wanghaoran/magformer/d2m2f_src"
sys.path.insert(0, SRC)
sys.path.insert(0, f"{SRC}/Mask2Former")

import register_32254_rgbd  # noqa: F401
from rgbd_concat_mapper import RGBDConcatMapper
from sgf_backbone import DepthTower

from detectron2.data import DatasetCatalog

import os as _os
RGB_MODE = "--rgb" in sys.argv
OUT_PATH = ("/home/hdd1/wanghaoran/magformer/pretrained/rgb_tower_distilled.pth" if RGB_MODE
            else "/home/hdd1/wanghaoran/magformer/pretrained/depth_tower_distilled.pth")


class TowerProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower = DepthTower(in_chans=3 if RGB_MODE else 1)
        self.heads = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in DepthTower.CHANS])

    def forward(self, depth):
        feats = self.tower(depth)
        return [h(f) for h, f in zip(self.heads, feats)]


class DepthMaskDataset(Dataset):
    def __init__(self, cfg):
        self.mapper = RGBDConcatMapper(cfg, is_train=True)
        self.items = DatasetCatalog.get("ecc20260318_1k_rgbd_train")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        d = self.mapper(self.items[idx])
        depth = d["image"][:3] if RGB_MODE else d["image"][3:4]  # raw 0-255 scale
        gt = d["instances"].gt_masks
        if hasattr(gt, "tensor"):
            g = gt.tensor
        else:
            g = torch.stack([torch.as_tensor(m) for m in gt], 0)
        union = (g.sum(dim=0) > 0).float().unsqueeze(0) if g.numel() else torch.zeros_like(depth[:1])
        return depth, union


def soft_dice(logit, target):
    p = torch.sigmoid(logit)
    num = 2 * (p * target).flatten(1).sum(1) + 1.0
    den = (p.flatten(1).sum(1) + target.flatten(1).sum(1)) + 1.0
    return (1 - num / den).mean()


def main():
    from detectron2.config import get_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = "bitmask"

    ds = DepthMaskDataset(cfg)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True)

    model = TowerProxy().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)
    scaler = torch.amp.GradScaler("cuda")

    it = 0
    while it < args.iters:
        for depth, union in dl:
            depth = depth.cuda(non_blocking=True)
            union = union.cuda(non_blocking=True)
            if RGB_MODE:
                mean = torch.tensor([123.675, 116.28, 103.53], device=depth.device).view(1, 3, 1, 1)
                std = torch.tensor([58.395, 57.12, 57.375], device=depth.device).view(1, 3, 1, 1)
                zn = (depth - mean) / std
            else:
                zn = (depth - 125.628) / 15.157  # what the backbone tower sees
            with torch.amp.autocast("cuda"):
                logits = model(zn)
                loss = 0.0
                for lg in logits:
                    tgt = F.interpolate(union, size=lg.shape[-2:], mode="nearest")
                    loss = loss + F.binary_cross_entropy_with_logits(lg, tgt) + soft_dice(lg.float(), tgt)
                loss = loss / len(logits)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1
            if it % 200 == 0:
                with torch.no_grad():
                    p = (torch.sigmoid(logits[-1].float()) > 0.5).float()
                    iou = (p * F.interpolate(union, size=p.shape[-2:], mode="nearest")).sum() / \
                          ((p + F.interpolate(union, size=p.shape[-2:], mode="nearest")).sum() - (p * F.interpolate(union, size=p.shape[-2:], mode="nearest")).sum() + 1e-6)
                print(f"iter {it} loss {loss.item():.4f} top-scale IoU {iou.item():.4f}", flush=True)
            if it >= args.iters:
                break

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save({"tower": model.tower.state_dict()}, OUT_PATH)
    print("saved", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
