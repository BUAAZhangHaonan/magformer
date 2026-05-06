# -*- coding: utf-8 -*-
"""Hungarian matcher with Mask2Former-compatible semantics.

Uses scipy's C-optimised Jonker-Volgenant solver on CPU for optimal
bipartite matching.  For MagFormer's typical cost matrices (100-200
queries x 5-50 targets) scipy runs in <0.1 ms per match — far faster
than any GPU-based approach due to serial-algorithm kernel-launch
overhead.  A pure-PyTorch JV fallback is included for environments
without scipy.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Pure-PyTorch JV fallback (float64 on CPU)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _lsa_pytorch(cost_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact LSA via Jonker-Volgenant on CPU (float64 for precision)."""
    n, m = cost_matrix.shape
    if n == 0 or m == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    device = cost_matrix.device
    cost = cost_matrix.double().cpu().contiguous()

    transposed = n > m
    if transposed:
        cost = cost.t().contiguous()
        n, m = m, n

    INF = cost.new_tensor(1e18)
    u = cost.new_zeros(n + 1)
    v = cost.new_zeros(m + 1)
    p = torch.zeros(m + 1, dtype=torch.long)
    way = torch.zeros(m + 1, dtype=torch.long)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = cost.new_full((m + 1,), INF)
        used = torch.zeros(m + 1, dtype=torch.bool)

        while True:
            used[j0] = True
            i0 = p[j0].item()
            reduced = cost[i0 - 1] - u[i0] - v[1:]
            unused = ~used[1:]
            better = unused & (reduced < minv[1:])
            minv[1:] = torch.where(better, reduced, minv[1:])
            way[1:] = torch.where(better, j0, way[1:])
            masked = torch.where(~used[1:], minv[1:], INF.expand(m))
            j1 = int(torch.argmin(masked)) + 1
            delta = minv[j1].item()
            used_f = used.double()
            v -= delta * used_f
            minv -= delta * (1.0 - used_f)
            used_idx = torch.where(used)[0]
            if used_idx.numel() > 0:
                rows = p[used_idx]
                u.scatter_add_(0, rows, cost.new_full((used_idx.numel(),), delta))
            j0 = j1
            if p[j0].item() == 0:
                break

        while True:
            j1 = way[j0].item()
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_list, col_list = [], []
    for j in range(1, m + 1):
        r = p[j].item()
        if r > 0:
            row_list.append(r - 1)
            col_list.append(j - 1)

    rows = torch.tensor(row_list, dtype=torch.long)
    cols = torch.tensor(col_list, dtype=torch.long)
    if transposed:
        rows, cols = cols, rows
    return rows.to(device), cols.to(device)


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

@torch.no_grad()
def linear_sum_assignment_gpu(cost_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve the linear sum assignment problem.

    Returns ``(row_ind, col_ind)`` — 1-D ``int64`` tensors on the **same
    device** as *cost_matrix*, each of length ``min(n, m)``.

    Uses scipy (fast C backend) when available; falls back to a pure
    PyTorch Jonker-Volgenant implementation otherwise.
    """
    if _HAS_SCIPY:
        device = cost_matrix.device
        C_cpu = cost_matrix.float().cpu().numpy()
        row_ind, col_ind = _scipy_lsa(C_cpu)
        return (
            torch.as_tensor(row_ind, dtype=torch.int64, device=device),
            torch.as_tensor(col_ind, dtype=torch.int64, device=device),
        )
    return _lsa_pytorch(cost_matrix)


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def _dice_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid()
    numerator = 2 * torch.matmul(inputs, targets.t())
    denom = inputs.sum(-1, keepdim=True) + targets.sum(-1).unsqueeze(0)
    cost = 1 - (numerator + 1) / (denom + 1)
    return cost


def _bce_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none")
    return (torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, 1 - targets)) / hw


def point_sample(input_tensor: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    grid = point_coords * 2.0 - 1.0
    grid = grid.unsqueeze(2)
    sampled = F.grid_sample(
        input_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled.squeeze(-1)


# ---------------------------------------------------------------------------
# HungarianMatcher
# ---------------------------------------------------------------------------

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

            row_ind, col_ind = linear_sum_assignment_gpu(C.float())
            indices.append((row_ind, col_ind))

        return indices
