#!/usr/bin/env python3.11
"""CPU-only aten-op census of the criterion forward (no GPU).

Counts every dispatched aten op for one criterion.forward with the
winners ON vs OFF on synthetic 66-GT / 200-query outputs. Op count is
the GIL/kernel-launch storm metric behind the 2026-09-21 perf incident.
"""
import sys
from collections import Counter

import torch

sys.path.insert(0, "/home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/source")

from magformer.models.common.criterion import SetCriterion  # noqa: E402
from magformer.models.common.matcher import HungarianMatcher  # noqa: E402

B, Q, NGT, H, W = 1, 200, 66, 512, 512
AUX = 8


class OpCounter(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()
        self.total = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        self.total += 1
        return func(*args, **(kwargs or {}))


def make_outputs(with_aux=True):
    torch.manual_seed(0)
    out = {
        "pred_logits": torch.randn(B, Q, 2),
        "pred_masks": torch.sigmoid(torch.randn(B, Q, H, W)) - 0.5,
        "pred_boxes": torch.rand(B, Q, 4),
    }
    if with_aux:
        out["aux_outputs"] = [
            {"pred_logits": torch.randn(B, Q, 2),
             "pred_masks": torch.sigmoid(torch.randn(B, Q, 256, 256)) - 0.5,
             "pred_boxes": torch.rand(B, Q, 4)}
            for _ in range(AUX)
        ]
    return out


def make_targets():
    torch.manual_seed(1)
    masks = torch.zeros(NGT, 1024, 1024)
    for i in range(NGT):
        y = (i * 37) % 900 + 40
        x = (i * 53) % 900 + 40
        s = 12 + (i % 5) * 10
        masks[i, y:y + s, x:x + s] = 1.0
    return [{"labels": torch.zeros(NGT, dtype=torch.long), "masks": masks}]


def build(winners: bool):
    matcher = HungarianMatcher(
        cost_class=2.0, cost_mask=5.0, cost_dice=1.0, num_points=12544,
        cost_bbox=5.0, cost_giou=2.0,
        small_gt_mode="aim" if winners else "uniform",
        small_gt_area=4096, small_gt_alpha=0.7,
    )
    return SetCriterion(
        num_classes=1, matcher=matcher,
        weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 1.0,
                     "loss_bbox": 5.0, "loss_giou": 2.0},
        eos_coef=0.1, losses=("labels", "masks", "boxes"),
        num_points=12544, dn_enabled=winners,
        mal_enabled=winners, bass_enabled=winners,
    )


def run(winners: bool, label: str):
    crit = build(winners)
    if winners:
        num_dn = 8 * 4
        outputs = make_outputs()
        outputs["pred_logits"] = torch.cat(
            [outputs["pred_logits"], torch.randn(B, num_dn, 2)], 1)
        outputs["pred_masks"] = torch.cat(
            [outputs["pred_masks"], torch.sigmoid(torch.randn(B, num_dn, H, H)) - 0.5], 1)
        outputs["pred_boxes"] = torch.cat(
            [outputs["pred_boxes"], torch.rand(B, num_dn, 4)], 1)
        outputs["dn_enabled"] = True
        outputs["dn_num_regular_queries"] = Q
        outputs["dn_meta"] = {
            "num_positives": 24, "num_negatives": 8, "dn_scalar": 3,
            "max_valid": 8, "batch_num_valid": [8],
            "gt_labels_padded": torch.zeros(B, 8, dtype=torch.long),
            "neg_valid": torch.ones(B, 8, dtype=torch.bool),
        }
        for a in outputs["aux_outputs"]:
            a["pred_logits"] = torch.cat([a["pred_logits"], torch.randn(B, num_dn, 2)], 1)
            a["pred_masks"] = torch.cat([a["pred_masks"], torch.sigmoid(torch.randn(B, num_dn, 256, 256)) - 0.5], 1)
            a["pred_boxes"] = torch.cat([a["pred_boxes"], torch.rand(B, num_dn, 4)], 1)
            a["dn_num_regular_queries"] = Q
    else:
        outputs = make_outputs()
    targets = make_targets()
    with OpCounter() as ctr:
        losses = crit(outputs, targets)
        float(losses["total_loss"])
    print(f"[{label}] total aten ops = {ctr.total}, loss = {float(losses['total_loss']):.3f}")
    for name, n in ctr.counts.most_common(14):
        print(f"   {n:6d}  {name[:70]}")
    return ctr.total


if __name__ == "__main__":
    torch.manual_seed(42)
    off = run(False, "winners OFF (A0-equivalent)")
    on = run(True, "winners ON  (B1 stack)")
    print(f"\ndelta = {on - off} ops ({(on/off - 1)*100:.0f}% growth)")
