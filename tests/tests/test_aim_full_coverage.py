"""AIM/grounded small-GT cost full-coverage tests (P1 bugfix, 2026-09-21).

Regression guard for the removed ``sel[:16]`` count cap: on images with more
than 16 eligible small GTs (train-set p90 is 41 per image), every eligible GT
must receive its own grounded/AIM cost column, and the chunked AIM
accumulation must be numerically identical to the unchunked reference.
"""

import math

import torch

from magformer.models.common.matcher import HungarianMatcher, point_sample


def _make_small_gts(n: int, h: int = 256, w: int = 256, seed: int = 0,
                    radius_range=(2, 5)) -> torch.Tensor:
    """Scatter ``n`` disjoint small disks (area < 4096 px^2 each)."""
    g = torch.Generator().manual_seed(seed)
    masks = torch.zeros(n, h, w)
    occupied = torch.zeros(h, w, dtype=torch.bool)
    yy, xx = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing="ij")
    placed = 0
    for flat in torch.randperm(h * w, generator=g)[: n * 40].tolist():
        cy_, cx_ = flat // w, flat % w
        r = int(torch.randint(radius_range[0], radius_range[1] + 1, (1,),
                              generator=g).item())
        disk = (yy - cy_) ** 2 + (xx - cx_) ** 2 <= r * r
        if disk.sum() == 0 or (disk & occupied).any():
            continue
        masks[placed] = disk.float()
        occupied |= disk
        placed += 1
        if placed == n:
            break
    assert placed == n, f"only placed {placed}/{n} disks"
    return masks


def _aim_reference(matcher: HungarianMatcher, out_mask_b: torch.Tensor,
                   tgt_masks: torch.Tensor):
    """Unchunked reference of the AIM cost math (same global R)."""
    device = out_mask_b.device
    H, W = tgt_masks.shape[-2:]
    m = tgt_masks.to(device) > 0.5
    areas = m.flatten(1).sum(dim=1)
    sel = ((areas < matcher.small_gt_area) & (areas > 0)
           ).nonzero().squeeze(1)
    any_row = m[sel].any(dim=2).float()
    any_col = m[sel].any(dim=1).float()
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
    ys = y0c[:, None] + base[None, :] * ((y1c - y0c)[:, None])
    xs = x0c[:, None] + base[None, :] * ((x1c - x0c)[:, None])
    gy = ys[:, :, None].expand(-1, R, R)
    gx = xs[:, None, :].expand(-1, R, R)
    coords = torch.stack(
        [gx / (W - 1), gy / (H - 1)], dim=-1).reshape(-1, R * R, 2)
    lab = point_sample(tgt_masks[sel].to(device)[:, None].float(),
                       coords).squeeze(1)
    probs = point_sample(
        out_mask_b[:, None].sigmoid(),
        coords.reshape(1, -1, 2).expand(out_mask_b.shape[0], -1, -1),
    ).squeeze(1).view(out_mask_b.shape[0], -1, R * R)
    A = lab.sum(dim=1)
    inter = torch.einsum("qnc,nc->qn", probs, lab)
    psum = probs.sum(dim=2)
    dice = 1.0 - 2.0 * inter / (psum + A[None, :]).clamp_min(1e-3)
    return sel, dice


def _mk(mode: str, **kw) -> HungarianMatcher:
    params = dict(cost_class=2.0, cost_mask=5.0, cost_dice=20.0,
                  num_points=12544, small_gt_points=0, small_gt_area=4096,
                  small_gt_alpha=0.7, small_gt_mode=mode, aim_w_bce=0.0)
    params.update(kw)
    return HungarianMatcher(**params)


def test_aim_covers_more_than_16_small_gts():
    torch.manual_seed(0)
    n_gt, n_q = 25, 32
    tgt_masks = _make_small_gts(n_gt)
    out_mask = torch.randn(n_q, 256, 256)
    matcher = _mk("aim")
    sel, bce, dice = matcher._aim_small_gt_costs(out_mask, tgt_masks)
    assert sel.numel() == n_gt, (
        f"AIM must cover all {n_gt} eligible GTs, got {sel.numel()} "
        "(regression of the historical [:16] cap)")
    assert dice.shape == (n_q, n_gt) and bce.shape == (n_q, n_gt)
    assert torch.all(bce == 0), "aim_w_bce=0 must yield zero BCE columns"


def test_aim_chunked_equals_unchunked_reference():
    torch.manual_seed(1)
    n_gt, n_q = 37, 24  # 37 = 16 + 16 + 5: crosses chunk boundaries
    tgt_masks = _make_small_gts(n_gt, seed=7)
    out_mask = torch.randn(n_q, 256, 256)
    matcher = _mk("aim")
    sel, _, dice = matcher._aim_small_gt_costs(out_mask, tgt_masks)
    ref_sel, ref_dice = _aim_reference(matcher, out_mask, tgt_masks)
    assert torch.equal(sel, ref_sel)
    assert torch.allclose(dice, ref_dice, atol=1e-6), (
        "chunked AIM accumulation must be numerically identical to the "
        f"unchunked reference (max diff {(dice - ref_dice).abs().max()})")


def test_aim_deterministic():
    torch.manual_seed(2)
    tgt_masks = _make_small_gts(20, seed=3)
    out_mask = torch.randn(16, 256, 256)
    matcher = _mk("aim")
    sel1, bce1, dice1 = matcher._aim_small_gt_costs(out_mask, tgt_masks)
    sel2, bce2, dice2 = matcher._aim_small_gt_costs(out_mask, tgt_masks)
    assert torch.equal(sel1, sel2)
    assert torch.equal(dice1, dice2) and torch.equal(bce1, bce2)


def test_aim_with_bce_weight_covers_all():
    torch.manual_seed(3)
    n_gt, n_q = 18, 12
    tgt_masks = _make_small_gts(n_gt, seed=11)
    out_mask = torch.randn(n_q, 256, 256)
    matcher = _mk("aim", aim_w_bce=0.5)
    sel, bce, dice = matcher._aim_small_gt_costs(out_mask, tgt_masks)
    assert sel.numel() == n_gt
    assert torch.isfinite(bce).all() and (bce != 0).any()


def test_grounded_covers_more_than_16_small_gts():
    torch.manual_seed(4)
    n_gt, n_q = 22, 16
    tgt_masks = _make_small_gts(n_gt, seed=5)
    out_mask = torch.randn(n_q, 256, 256)
    matcher = _mk("grounded", small_gt_points=96)
    res = matcher._grounded_small_gt_costs(out_mask, tgt_masks)
    assert res is not None
    kept_t, bce, dice = res
    assert kept_t.numel() == n_gt, (
        f"grounded path must cover all {n_gt} eligible GTs, got "
        f"{kept_t.numel()} (regression of the historical [:16] cap)")
    assert bce.shape == (n_q, n_gt) and dice.shape == (n_q, n_gt)
    assert torch.isfinite(bce).all() and torch.isfinite(dice).all()


def test_forward_runs_with_dense_small_gts():
    """End-to-end forward with 25 small GTs assigns every GT to a query."""
    torch.manual_seed(5)
    n_gt, n_q = 25, 40
    tgt_masks = _make_small_gts(n_gt, seed=13)
    outputs = {
        "pred_logits": torch.randn(1, n_q, 2),
        "pred_masks": torch.randn(1, n_q, 256, 256),
        "pred_boxes": torch.rand(1, n_q, 4) * 0.2 + 0.3,
    }
    boxes = []
    for g in range(n_gt):
        nz = tgt_masks[g].nonzero()
        boxes.append([nz[:, 1].min() / 256, nz[:, 0].min() / 256,
                      nz[:, 1].max() / 256, nz[:, 0].max() / 256])
    targets = [{
        "labels": torch.zeros(n_gt, dtype=torch.long),
        "masks": tgt_masks,
        "boxes": torch.tensor(boxes, dtype=torch.float32),
    }]
    matcher = _mk("aim", cost_bbox=5.0, cost_giou=2.0)
    indices = matcher(outputs, targets)
    assert len(indices) == 1
    src, tgt = indices[0]
    assert tgt.numel() == n_gt, "every GT must be assigned in forward"
    assert len(set(tgt.tolist())) == n_gt
