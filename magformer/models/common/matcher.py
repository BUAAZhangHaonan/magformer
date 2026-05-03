# -*- coding: utf-8 -*-
"""Hungarian matcher with Mask2Former-compatible semantics."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _dice_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute pairwise dice cost between inputs (Q, P) and targets (T, P)."""
    inputs = inputs.sigmoid()
    numerator = 2 * torch.matmul(inputs, targets.t())
    denom = inputs.sum(-1, keepdim=True) + targets.sum(-1).unsqueeze(0)
    cost = 1 - (numerator + 1) / (denom + 1)
    return cost


def _bce_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute pairwise BCE cost between inputs (Q, P) and targets (T, P)."""
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none")
    return (torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, 1 - targets)) / hw


def point_sample(input_tensor: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    """
    Sample points from feature maps.

    Args:
        input_tensor: (N, C, H, W)
        point_coords: (N, P, 2) in [0, 1], order (x, y)
    Returns:
        (N, C, P)
    """
    grid = point_coords * 2.0 - 1.0
    grid = grid.unsqueeze(2)
    sampled = F.grid_sample(
        input_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled.squeeze(-1)


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
        num_points: int = 12544,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, outputs: dict, targets: List[dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # Keep matching costs in fp32 even under AMP.
        # In fp16, summing BCE over many sampled points (e.g. 65536) may overflow to +inf.
        out_logits = outputs["pred_logits"].float()
        out_masks = outputs["pred_masks"].float()

        indices = []
        for b in range(bs):
            tgt_masks = targets[b]["masks"].float()
            tgt_labels = targets[b]["labels"].long()
            if tgt_masks.numel() == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64, device=out_masks.device),
                    torch.empty(0, dtype=torch.int64, device=out_masks.device),
                ))
                continue

            out_prob = out_logits[b].softmax(-1)
            cost_class = -out_prob[:, tgt_labels]

            out_mask = out_masks[b][:, None]
            tgt_mask = tgt_masks[:, None].to(
                device=out_mask.device, dtype=out_mask.dtype)

            point_coords = torch.rand(
                1, self.num_points, 2, device=out_mask.device, dtype=out_mask.dtype
            )
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
            ).squeeze(1)
            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
            ).squeeze(1)

            cost_mask = _bce_cost(out_mask, tgt_mask)
            cost_dice = _dice_cost(out_mask, tgt_mask)

            C = self.cost_class * cost_class + self.cost_mask * \
                cost_mask + self.cost_dice * cost_dice
            if not torch.isfinite(C).all():
                C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)
            C = C.float().cpu()

            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64,
                                device=out_masks.device),
                torch.as_tensor(col_ind, dtype=torch.int64,
                                device=out_masks.device),
            ))

        return indices
