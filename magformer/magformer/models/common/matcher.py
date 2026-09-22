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

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, masks_to_boxes_cxcywh

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
        scale_balanced: bool = False,
        cost_bbox: float = 0.0,
        cost_giou: float = 0.0,
        small_gt_points: int = 0,
        small_gt_area: int = 4096,
        small_gt_alpha: float = 0.7,
        small_gt_mode: str = "grounded",
        aim_w_bce: float = 0.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
        self.scale_balanced = scale_balanced
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # Small-GT grounded cost points (aps_20260913): for GT below
        # small_gt_area px^2 (at the 1024^2 target resolution), the uniform
        # point set carries ~E[points] = 12544*area/1024^2 foreground samples
        # (~4.8 for a 400px^2 GT), making the mask-cost column noise-dominated.
        # Drawing small_gt_points per small GT (2/3 from the GT foreground,
        # 1/3 from its bbox+30% margin) restores signal; blended into the
        # uniform column at small_gt_alpha. Class/box cost rows unchanged.
        self.small_gt_points = int(small_gt_points)
        self.small_gt_area = int(small_gt_area)
        self.small_gt_alpha = float(small_gt_alpha)
        # small_gt_mode "aim" (arena 2026-09-17 winner): replaces the sampled
        # grounded points with an exact soft-Dice (+ optional balanced BCE x
        # aim_w_bce) computed deterministically on each small GT's bbox+30%
        # window — zero RNG, zero sampling variance, scale-free statistic.
        # small_gt_points is ignored in this mode. Lab: matched-IoU 0.4134 vs
        # uniform 0.4043 (oracle ceiling 0.4152), margin 2.43 sigma vs 0.13.
        self.small_gt_mode = str(small_gt_mode)
        self.aim_w_bce = float(aim_w_bce)
        # Chunk size for the AIM cost accumulation loop. Bounds the
        # (Q, chunk, R*R) point_sample activation so images with many small
        # GTs (train p90 = 41 eligible per image) cannot blow memory; the
        # window-grid resolution R stays global across chunks, so chunking
        # does not change the computed costs.
        self.aim_gt_chunk = 16

    @torch.no_grad()
    def _grounded_small_gt_costs(self, out_mask_b: torch.Tensor, tgt_masks: torch.Tensor):
        """Grounded mask costs for small-GT columns.

        For each GT with 0 < area < small_gt_area (px^2 at target resolution),
        draws small_gt_points coordinates: 2/3 from the GT foreground pixels,
        1/3 uniform in its bbox dilated by 30% (min 3px). All queries are
        sampled with a single point_sample over the concatenated coordinates.
        Returns (kept_gt_indices, bce_cost (Q, n_kept), dice_cost (Q, n_kept)),
        or None when no small GT is present.
        """
        device = out_mask_b.device
        H, W = tgt_masks.shape[-2:]
        areas = tgt_masks.flatten(1).sum(dim=1)
        sel = ((areas < self.small_gt_area) & (areas > 0)).nonzero().squeeze(1)
        if sel.numel() == 0:
            return None
        # No count cap: every eligible small GT gets a grounded column (the
        # per-GT loop below is memory-flat; the historical [:16] cap silently
        # dropped ~half of the eligible GTs on dense images).
        P = self.small_gt_points
        n_fg = int(P * 2 / 3)
        n_bg = P - n_fg
        coords_list = []
        kept = []
        for g in sel.tolist():
            nz = tgt_masks[g].nonzero()
            if nz.shape[0] == 0:
                continue
            fg_idx = torch.randint(nz.shape[0], (n_fg,), device=device)
            fg = nz[fg_idx].float()
            y0 = nz[:, 0].min().to(device).float()
            y1 = nz[:, 0].max().to(device).float()
            x0 = nz[:, 1].min().to(device).float()
            x1 = nz[:, 1].max().to(device).float()
            my = max(3.0, 0.3 * float(y1 - y0 + 1))
            mx = max(3.0, 0.3 * float(x1 - x0 + 1))
            y0c = min(max(float(y0 - my), 0.0), float(H - 1))
            y1c = min(max(float(y1 + my), 0.0), float(H - 1))
            x0c = min(max(float(x0 - mx), 0.0), float(W - 1))
            x1c = min(max(float(x1 + mx), 0.0), float(W - 1))
            bg_y = torch.rand(n_bg, device=device) * (y1c - y0c) + y0c
            bg_x = torch.rand(n_bg, device=device) * (x1c - x0c) + x0c
            ys = torch.cat([fg[:, 0], bg_y])
            xs = torch.cat([fg[:, 1], bg_x])
            # point_sample expects (x, y) normalized coords
            coords_list.append(torch.stack([xs / (W - 1), ys / (H - 1)], dim=-1))
            kept.append(g)
        if not kept:
            return None
        coords_cat = torch.stack(coords_list, dim=0)  # (n_kept, P, 2)
        n_kept = len(kept)
        tgt_labels = point_sample(
            tgt_masks[kept][:, None].float(), coords_cat
        ).squeeze(1)  # (n_kept, P)
        out_pts = point_sample(
            out_mask_b[:, None],
            coords_cat.reshape(1, -1, 2).expand(out_mask_b.shape[0], -1, -1),
        ).squeeze(1).view(out_mask_b.shape[0], n_kept, P)  # (Q, n_kept, P)
        bce_cols = []
        dice_cols = []
        for j in range(n_kept):
            bce_cols.append(_bce_cost(out_pts[:, j, :], tgt_labels[j:j + 1]))
            dice_cols.append(_dice_cost(out_pts[:, j, :], tgt_labels[j:j + 1].float()))
        kept_t = torch.as_tensor(kept, dtype=torch.long, device=device)
        return kept_t, torch.cat(bce_cols, dim=1), torch.cat(dice_cols, dim=1)

    @torch.no_grad()
    def _aim_small_gt_costs(self, out_mask_b: torch.Tensor, tgt_masks: torch.Tensor):
        """AIM small-GT columns (arena design C, 2026-09-17 winner).

        For each GT with 0 < area < small_gt_area, computes soft-Dice exactly
        on a deterministic R x R grid over its bbox dilated by 30% (min 3px)
        — no point sampling, no RNG, so the column is invariant to resampling
        and the statistic is scale-free (denominator normalizes object size).
        Returns (kept_gt_indices, bce (Q, n_kept), dice (Q, n_kept)) with the
        same contract as _grounded_small_gt_costs; bce is zero unless
        aim_w_bce > 0 (lab w-grid: any BCE weight hurt ranking).
        """
        device = out_mask_b.device
        tgt_masks = tgt_masks.to(device=device)
        H, W = tgt_masks.shape[-2:]
        m = tgt_masks > 0.5
        areas = m.flatten(1).sum(dim=1)
        sel = ((areas < self.small_gt_area) & (areas > 0)).nonzero().squeeze(1)
        if sel.numel() == 0:
            return None

        # vectorized per-GT bbox: first/last true row/col (global over sel,
        # so the window grid resolution R below is shared by every chunk —
        # chunking therefore does not change the computed costs)
        any_row = m[sel].any(dim=2).float()          # (n, H)
        any_col = m[sel].any(dim=1).float()          # (n, W)
        y0 = any_row.argmax(dim=1)
        y1 = H - 1 - any_row.flip(dims=[1]).argmax(dim=1)
        x0 = any_col.argmax(dim=1)
        x1 = W - 1 - any_col.flip(dims=[1]).argmax(dim=1)
        ext = torch.stack([y1 - y0 + 1.0, x1 - x0 + 1.0], dim=1)
        marg = torch.clamp(0.3 * ext, min=3.0)
        y0c = (y0.float() - marg[:, 0]).clamp(0, H - 1)
        y1c = (y1.float() + marg[:, 0]).clamp(0, H - 1)
        x0c = (x0.float() - marg[:, 1]).clamp(0, W - 1)
        x1c = (x1.float() + marg[:, 1]).clamp(0, W - 1)
        R = int(torch.clamp((y1c - y0c).maximum(x1c - x0c).max().round() + 2,
                            min=8.0, max=64.0).item())

        base = torch.linspace(0.0, 1.0, R, device=device)
        ys = y0c[:, None] + base[None, :] * ((y1c - y0c)[:, None])   # (n, R)
        xs = x0c[:, None] + base[None, :] * ((x1c - x0c)[:, None])   # (n, R)
        gy = ys[:, :, None].expand(-1, R, R)                        # (n, R, R)
        gx = xs[:, None, :].expand(-1, R, R)                        # (n, R, R)
        coords = torch.stack(
            [gx / (W - 1), gy / (H - 1)], dim=-1).reshape(-1, R * R, 2)  # (n,P,2) (x,y)

        # No count cap: every eligible small GT gets its AIM column (the
        # historical [:16] cap silently fell ~50% of eligible GTs back to the
        # noise-dominated uniform columns on dense images). Accumulation is
        # chunked purely to bound the (Q, chunk, R*R) activation.
        bce_cols = []
        dice_cols = []
        for s0 in range(0, sel.numel(), self.aim_gt_chunk):
            sel_c = sel[s0:s0 + self.aim_gt_chunk]
            coords_c = coords[s0:s0 + self.aim_gt_chunk]
            lab = point_sample(
                tgt_masks[sel_c][:, None].float(), coords_c).squeeze(1)  # (n_c, P)
            probs = point_sample(
                out_mask_b[:, None].sigmoid(),
                coords_c.reshape(1, -1, 2).expand(out_mask_b.shape[0], -1, -1),
            ).squeeze(1).view(out_mask_b.shape[0], -1, R * R)           # (Q, n_c, P)

            A = lab.sum(dim=1)                                          # (n_c,)
            inter = torch.einsum("qnc,nc->qn", probs, lab)              # (Q, n_c)
            psum = probs.sum(dim=2)                                     # (Q, n_c)
            dice_cols.append(
                1.0 - 2.0 * inter / (psum + A[None, :]).clamp_min(1e-3))

            if self.aim_w_bce > 0:
                p = probs.clamp(1e-4, 1 - 1e-4)
                fg = -(torch.log(p) * lab[None]).sum(2) / A.clamp_min(1.0)[None]
                bg = -(torch.log1p(-p) * (1 - lab)[None]).sum(2) \
                    / (R * R - A).clamp_min(1.0)[None]
                bce_cols.append(self.aim_w_bce * 0.5 * (fg + bg))
            else:
                bce_cols.append(torch.zeros(
                    out_mask_b.shape[0], sel_c.numel(),
                    device=device, dtype=dice_cols[-1].dtype))
        return sel, torch.cat(bce_cols, dim=1), torch.cat(dice_cols, dim=1)

    @torch.no_grad()
    def forward(self, outputs: dict, targets: List[dict],
                return_quality: bool = False):
        """Match queries to GTs (Hungarian, AIM-blended small-GT columns).

        ``return_quality=True`` additionally returns per-image quality vectors
        q = soft-Dice of each ASSIGNED pair (AIM deterministic columns for
        small GTs, uniform-point columns otherwise) — the MAL-CP+ matchability
        target (arena P4-c winner): the matcher selected the pair by this
        statistic, so the classification head learns to regress it. Both the
        assignment itself and the no-quality path are bit-identical to the
        previous behaviour.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_logits = outputs["pred_logits"].float()
        out_masks = outputs["pred_masks"].float()

        indices = []
        qualities = []
        for b in range(bs):
            tgt_masks = targets[b]["masks"].float()
            tgt_labels = targets[b]["labels"].long()
            if tgt_masks.numel() == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64, device=out_masks.device),
                    torch.empty(0, dtype=torch.int64, device=out_masks.device),
                ))
                if return_quality:
                    qualities.append(torch.empty(0, device=out_masks.device))
                continue

            out_prob = out_logits[b].sigmoid()  # sigmoid 不是 softmax
            # tgt_labels 范围 [0, num_classes-1]，eos 不参与 matcher cost
            cost_class = -out_prob[:, tgt_labels.clamp(max=out_logits.shape[-1] - 1)]

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
            # MAL q-target source: defaults to the uniform soft-Dice cost;
            # the small-GT branch below overwrites the AIM columns. Hoisted
            # out of the branch -- return_quality readers crashed with
            # UnboundLocalError whenever mal_enabled ran with the default
            # matcher (no small-GT costs).
            quality_dice = cost_dice

            C = self.cost_class * cost_class + self.cost_mask * cost_mask + self.cost_dice * cost_dice

            # Box cost (L1 + GIoU) for hybrid matching
            if self.cost_bbox > 0 or self.cost_giou > 0:
                if "pred_boxes" not in outputs:
                    raise KeyError(
                        "pred_boxes are required when cost_bbox or cost_giou is enabled"
                    )
                masks_b = tgt_masks
                pred_boxes_b = outputs["pred_boxes"][b]
                gt_boxes = masks_to_boxes_cxcywh(masks_b).to(pred_boxes_b)

                if self.cost_bbox > 0:
                    cost_bbox_b = torch.cdist(pred_boxes_b, gt_boxes, p=1)
                else:
                    cost_bbox_b = torch.zeros(num_queries, masks_b.shape[0], device=masks_b.device)

                if self.cost_giou > 0:
                    cost_giou_b = -generalized_box_iou(
                        box_cxcywh_to_xyxy(pred_boxes_b),
                        box_cxcywh_to_xyxy(gt_boxes),
                    )
                else:
                    cost_giou_b = torch.zeros(num_queries, masks_b.shape[0], device=masks_b.device)

                C = C + self.cost_bbox * cost_bbox_b + self.cost_giou * cost_giou_b

            # Blend grounded point costs into small-GT columns (class/box rows
            # and large-GT columns untouched): raises the mask-cost SNR for the
            # size class where the uniform 12544-point set is noise-dominated.
            if self.small_gt_points > 0 or self.small_gt_mode == "aim":
                if self.small_gt_mode == "aim":
                    grounded = self._aim_small_gt_costs(out_masks[b], tgt_masks)
                else:
                    grounded = self._grounded_small_gt_costs(out_masks[b], tgt_masks)
                quality_dice = cost_dice  # (Q, n) uniform soft-Dice cost
                if grounded is not None:
                    kept_t, bce_g, dice_g = grounded
                    mask_u = self.cost_mask * cost_mask[:, kept_t] + self.cost_dice * cost_dice[:, kept_t]
                    mask_g = self.cost_mask * bce_g + self.cost_dice * dice_g
                    C = C.index_copy(
                        1, kept_t,
                        C[:, kept_t] + self.small_gt_alpha * (mask_g - mask_u),
                    )
                    # MAL quality: small GTs read the deterministic AIM dice
                    quality_dice = quality_dice.index_copy(1, kept_t, dice_g)

            if not torch.isfinite(C).all():
                C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)

            if self.scale_balanced:
                gt_areas = tgt_masks.flatten(1).sum(dim=1)  # (num_gt,)
                mean_area = gt_areas.mean()
                area_weight = (mean_area / (gt_areas + 1.0)).sqrt()
                area_weight = area_weight / area_weight.mean()
                C = C * area_weight.unsqueeze(0)

            row_ind, col_ind = linear_sum_assignment_gpu(C.float())
            if return_quality:
                # q = soft-Dice of the assigned pair (cost form 1 - 2I/(P+A))
                qualities.append(
                    (1.0 - quality_dice[row_ind, col_ind]).clamp(0.0, 1.0))
            indices.append((row_ind, col_ind))

        if return_quality:
            return indices, qualities
        return indices

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"cost_class={self.cost_class}, "
            f"cost_mask={self.cost_mask}, "
            f"cost_dice={self.cost_dice}, "
            f"num_points={self.num_points}, "
            f"scale_balanced={self.scale_balanced}, "
            f"cost_bbox={self.cost_bbox}, "
            f"cost_giou={self.cost_giou})"
        )
