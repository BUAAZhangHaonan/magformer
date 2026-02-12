# -*- coding: utf-8 -*-
"""
Hungarian Matcher

Minimal matching for single-class instance segmentation.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _dice_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute pairwise dice cost between inputs (Q, HW) and targets (T, HW)."""
    inputs = inputs.sigmoid()
    numerator = 2 * torch.matmul(inputs, targets.t())
    denom = inputs.sum(-1, keepdim=True) + targets.sum(-1).unsqueeze(0)
    cost = 1 - (numerator + 1) / (denom + 1)
    return cost


def _bce_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute pairwise BCE cost between inputs (Q, HW) and targets (T, HW). Returns (Q, T)."""
    inputs = inputs.sigmoid().clamp(1e-6, 1 - 1e-6)
    pos = torch.matmul(inputs.log(), targets.t())       # (Q, T)
    neg = torch.matmul((1 - inputs).log(), (1 - targets).t())  # (Q, T)
    cost = -(pos + neg) / inputs.shape[1]
    return cost


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, outputs: dict, targets: List[dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_logits = outputs["pred_logits"].squeeze(-1)
        out_masks = outputs["pred_masks"]

        indices = []
        for b in range(bs):
            tgt_masks = targets[b]["masks"].float()
            if tgt_masks.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            if tgt_masks.shape[-2:] != out_masks.shape[-2:]:
                tgt_masks = F.interpolate(
                    tgt_masks[:, None], size=out_masks.shape[-2:], mode="nearest"
                )[:, 0]

            out_mask = out_masks[b].flatten(1)
            tgt_mask = tgt_masks.flatten(1)

            cost_class = -out_logits[b][:, None].repeat(1, tgt_mask.shape[0])
            cost_mask = _bce_cost(out_mask, tgt_mask)
            cost_dice = _dice_cost(out_mask, tgt_mask)

            C = self.cost_class * cost_class + self.cost_mask * cost_mask + self.cost_dice * cost_dice
            C = C.cpu()

            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))

        return indices
