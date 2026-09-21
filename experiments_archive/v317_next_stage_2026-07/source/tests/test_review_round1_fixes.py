"""Review round-1 confirmed-bug regression tests (2026-09-22).

One test per confirmed finding from the three-reviewer deep review:
R1 (principle): bass sampler off-by-one; DN QFL un-detached target;
    DN noise iid-absolute -> box-relative ladder; DN GT selection.
R2 (logic/impl): DN negative xyxy/cxcywh frame confusion; matcher
    quality_dice UnboundLocalError; EMA apply_shadow stale non-float
    buffers; _seed_calls advancing on eval forwards.
R3 (efficiency): loader snapshot cadence helper.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from magformer.models.magformer.arch import MagFormerArch
from magformer.models.common.box_ops import box_cxcywh_to_xyxy
from magformer.models.common.criterion import SetCriterion
from magformer.models.common.matcher import HungarianMatcher


# ---------------------------------------------------------------- fixtures

def _dn_stub(cap=8, scalar=3, hidden=32, num_queries=64, ladder=None,
             small_area=4096.0):
    return SimpleNamespace(
        dn_enabled=True, training=True,
        dn_scalar=scalar, dn_box_noise_scale=0.3, dn_max_gt_cap=cap,
        dn_noise_ladder=ladder, dn_small_gt_area=small_area,
        num_classes=1,
        query_embed=SimpleNamespace(
            embedding_dim=hidden,
            weight=torch.nn.Parameter(torch.randn(num_queries, hidden))),
        _boxes_from_masks=MagFormerArch._boxes_from_masks,
        _sinusoidal_box_pe=MagFormerArch._sinusoidal_box_pe,
    )


def _gt_masks_with_sizes(sizes, H, W, ny, nx, ygap, xgap, y0=10, x0=10):
    """One image whose GT g is a sizes[g]-px square on a ny x nx grid.
    Layout must keep every box strictly inside the canvas: clipping a GT
    silently changes its area and breaks selection assertions."""
    max_gt = max(len(sizes), 1)
    masks = torch.zeros(1, max_gt, H, W)
    pad = torch.zeros(1, max_gt, dtype=torch.bool)
    for g, s in enumerate(sizes):
        y = y0 + (g // nx) * ygap
        x = x0 + (g % nx) * xgap
        assert y + s <= H and x + s <= W, (
            f"fixture layout clips GT {g} ({s}px at y={y},x={x}, canvas "
            f"{H}x{W})")
        masks[0, g, y:y + s, x:x + s] = 1.0
    pad[0, :len(sizes)] = True
    return masks, pad


# ------------------------------------------------- R1#2: bass off-by-one

def test_bass_sampler_reaches_last_pool_cell():
    """floor(r*(L-1)) never sampled the final cell of any segment; the fix
    must give a discrete uniform over {0..L-1} (statistical check)."""
    torch.manual_seed(0)
    pool_cells = torch.arange(10)
    offsets = torch.tensor([0, 3, 10])  # pool0 = cells {0,1,2}
    gt_ids = torch.zeros(4096, dtype=torch.long)
    coords = SetCriterion._bass_sample_pool(
        SimpleNamespace(_bass_pack_w=10, _bass_pack_h=1),
        pool_cells, offsets, gt_ids, k=256)
    # pool0 has L=3 cells -> x coords take 3 distinct values; the previous
    # implementation could only ever produce 2 of them
    xs = coords[..., 0].reshape(-1)
    levels = torch.unique((xs * 9).round() / 9)
    assert len(levels) == 3, f"expected all 3 pool cells hit, got {levels}"


# --------------------------------------------- R1#3: DN QFL target detach

def test_dn_qfl_target_is_detached():
    """q_full must be a constant target: the classification term may not
    push gradients into the mask head through the dice it regresses."""
    torch.manual_seed(1)
    B, num_regular, cap, scalar = 1, 6, 4, 2
    num_pos, num_neg = cap * scalar, cap
    num_dn = num_pos + num_neg
    H = W = 64
    stub = SimpleNamespace(
        num_classes=1, num_points=12544, dn_small_gt_area=4096.0,
        mal_enabled=True, mal_beta=2.0)
    logits = torch.randn(B, num_regular + num_dn, 2)
    masks_pred = torch.randn(B, num_regular + num_dn, H, W,
                             requires_grad=True)
    outputs = {
        "pred_logits": logits, "pred_masks": masks_pred,
        "dn_num_regular_queries": num_regular,
    }
    targets = [{"masks": torch.zeros(4, H, W),
                "labels": torch.zeros(4, dtype=torch.long)}]
    targets[0]["masks"][0, 10:40, 10:40] = 1.0
    targets[0]["masks"][1, 50:60, 50:60] = 1.0
    neg_valid = torch.zeros(B, cap, dtype=torch.bool)
    neg_valid[0, 0] = True
    dn_meta = {
        "num_positives": num_pos, "num_negatives": num_neg,
        "dn_scalar": scalar, "max_valid": cap,
        "batch_num_valid": [2],
        "gt_labels_padded": torch.zeros(B, cap, dtype=torch.long),
        "gt_pad_mask": None, "neg_valid": neg_valid, "num_dn": num_dn,
    }
    losses = SetCriterion._compute_dn_loss(stub, outputs, targets, dn_meta)
    if "loss_dn_ce" in losses and losses["loss_dn_ce"].requires_grad:
        losses["loss_dn_ce"].backward(retain_graph=True)
        assert masks_pred.grad is None or float(
            masks_pred.grad.abs().sum()) == 0.0, (
            "DN QFL leaked classification gradient into the mask head")


# ------------------------------- R2#1 / R1#1: DN negative frame correctness

def test_dn_negatives_are_valid_cxcywh_off_all_gts():
    """The old xyxy-formulas-on-cxcywh produced inverted candidate boxes and
    a vacuous IoU gate. Valid negatives must convert to proper xyxy boxes
    and hold max IoU < 0.1 against every GT of the image."""
    torch.manual_seed(2)
    stub = _dn_stub(cap=4, scalar=1)
    # three well-separated 20px boxes; annotation order shuffled vs size
    H = W = 200
    masks = torch.zeros(1, 3, H, W)
    for g, (y, x) in enumerate([(20, 20), (20, 100), (100, 20)]):
        masks[0, g, y:y + 20, x:x + 20] = 1.0
    pad = torch.tensor([[True, True, True]])
    labels = torch.zeros(1, 3, dtype=torch.long)
    _, _, meta = MagFormerArch._generate_dn_queries(
        stub, None, None, labels, masks, pad)
    neg_boxes = meta["neg_boxes"][0]          # (cap, 4) cxcywh
    neg_valid = meta["neg_valid"][0]
    nv = meta["batch_num_valid"][0]
    assert bool(neg_valid[:nv].all()), "all negatives should place on empty canvas"
    assert not bool(neg_valid[nv:].any()), "padded slots must stay invalid"
    xy = box_cxcywh_to_xyxy(neg_boxes[:nv])
    assert (xy[:, 2] > xy[:, 0]).all() and (xy[:, 3] > xy[:, 1]).all(), (
        "inverted negative box: frame confusion regressed")
    gt_boxes = torch.stack([
        MagFormerArch._boxes_from_masks(masks[0, g][None])[0]
        for g in range(nv)])
    gt_xy = box_cxcywh_to_xyxy(gt_boxes)
    lt = torch.maximum(xy[:, None, :2], gt_xy[None, :, :2])
    rb = torch.minimum(xy[:, None, 2:], gt_xy[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=2)
    area = ((xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1]))[:, None] * \
        ((gt_xy[:, 2] - gt_xy[:, 0]) * (gt_xy[:, 3] - gt_xy[:, 1]))[None, :]
    iou = inter / (area - inter + 1e-6)
    assert float(iou.max()) < 0.1, (
        f"negative landed on a real GT: max IoU {float(iou.max()):.3f}")


# ------------------------------------- R1#4/verdict: noise ladder + selection

def test_dn_positive_noise_is_box_relative_ladder():
    torch.manual_seed(3)
    ladder = [0.05, 0.15, 0.30]
    stub = _dn_stub(cap=2, scalar=3, ladder=ladder)
    sizes = [48, 48]
    masks, pad = _gt_masks_with_sizes(sizes, H=256, W=256, ny=1, nx=3,
                                       ygap=0, xgap=40)
    labels = torch.zeros_like(pad, dtype=torch.long)
    _, _, meta = MagFormerArch._generate_dn_queries(
        stub, None, None, labels, masks, pad)
    gt = meta["gt_boxes_padded"][0]           # (cap, 4) cxcywh
    scalar = stub.dn_scalar
    cap = stub.dn_max_gt_cap
    # re-derive the noisy positives through a capture of the PE input
    captured = {}

    def _spy(boxes, dim):
        captured.setdefault("boxes", []).append(boxes.detach().clone())
        return torch.randn(boxes.shape[0], boxes.shape[1], dim)

    stub._sinusoidal_box_pe = staticmethod(_spy)
    MagFormerArch._generate_dn_queries(stub, None, None, labels, masks, pad)
    pos = captured["boxes"][0].reshape(cap, scalar, 4)   # cxcywh
    for g in range(cap):
        for s, tier in enumerate(ladder):
            dxy = (pos[g, s, :2] - gt[g, :2]).abs()
            wh = gt[g, 2:4]
            # center shift bound: tier * (w,h)/2 (+clamp slack)
            assert bool((dxy <= tier * wh / 2 + 1e-4).all()), (
                f"gt{g} tier{s}: displacement {dxy.tolist()} exceeds "
                f"{tier}*wh/2 for box {wh.tolist()}")
            ratio = pos[g, s, 2:4] / wh
            assert bool((ratio >= 1 - tier - 1e-4).all()), (
                f"gt{g} tier{s}: wh shrank below (1-tier): {ratio.tolist()}")
            assert bool((ratio <= 1 + tier + 1e-4).all()), (
                f"gt{g} tier{s}: wh grew above (1+tier): {ratio.tolist()}")


def test_dn_selection_smallest_floor_plus_aim_eligible():
    """10 GTs on a 256px canvas, 7 AIM-eligible (area<=4096px^2): selection
    must be exactly those 7 eligible ones (the 4 smallest among them satisfy
    the floor) — the three 70px GTs (4900px^2, over the line) stay out —
    NOT the first 8 by annotation order."""
    torch.manual_seed(4)
    stub = _dn_stub(cap=8, scalar=3, small_area=4096.0)
    sizes = [70, 48, 8, 70, 40, 16, 24, 70, 32, 48]
    masks, pad = _gt_masks_with_sizes(sizes, H=400, W=400, ny=2, nx=5,
                                       ygap=200, xgap=78)
    labels = torch.zeros_like(pad, dtype=torch.long)
    _, _, meta = MagFormerArch._generate_dn_queries(
        stub, None, None, labels, masks, pad)
    assert meta["batch_num_valid"][0] == 7, (
        "7 eligible GTs exist; the cap is not reached and ineligible GTs "
        "must not backfill beyond the smallest-4 floor")
    areas = (meta["gt_boxes_padded"][0][:meta["batch_num_valid"][0], 2] *
             meta["gt_boxes_padded"][0][:meta["batch_num_valid"][0], 3]
             * 400 * 400)
    areas_px = sorted(int(a) for a in areas.tolist())
    expected = sorted(s * s for s in sizes if s * s <= 4096)
    assert len(areas_px) == len(expected)
    for got, exp in zip(areas_px, expected):
        assert got == pytest.approx(exp, abs=4), (
            f"selection not area-preferring: got {areas_px}, "
            f"expected {expected}")
    assert all(a <= 4096 for a in areas_px), (
        "ineligible (over-line) GT entered without a floor slot")


def test_dn_selection_floor_when_no_eligible():
    """All GTs over the eligibility line: still denoise the 4 smallest."""
    torch.manual_seed(5)
    stub = _dn_stub(cap=8, scalar=2, small_area=1000.0)
    sizes = [80, 90, 70, 85, 75, 95]
    masks, pad = _gt_masks_with_sizes(sizes, H=320, W=320, ny=2, nx=3,
                                       ygap=150, xgap=105)
    labels = torch.zeros_like(pad, dtype=torch.long)
    _, _, meta = MagFormerArch._generate_dn_queries(
        stub, None, None, labels, masks, pad)
    assert meta["batch_num_valid"][0] == 4, "floor is exactly 4"
    areas = (meta["gt_boxes_padded"][0][:meta["batch_num_valid"][0], 2] *
             meta["gt_boxes_padded"][0][:meta["batch_num_valid"][0], 3]
             * 320 * 320).tolist()
    assert sorted(int(a) for a in areas) == sorted(
        [s * s for s in [70, 75, 80, 85]])


# ------------------------------------- R2#5: matcher quality_dice hoist

def test_matcher_return_quality_without_small_gt_config():
    """mal_enabled + default matcher (no small-GT costs) used to raise
    UnboundLocalError on quality_dice; the hoist must make it work."""
    torch.manual_seed(6)
    m = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0)
    assert m.small_gt_points == 0 and m.small_gt_mode != "aim"
    B, Q, H, W = 1, 4, 16, 16
    outputs = {
        "pred_logits": torch.randn(B, Q, 2),
        "pred_masks": torch.randn(B, Q, H, W),
    }
    targets = [{
        "masks": (torch.rand(2, H, W) > 0.7).float(),
        "labels": torch.zeros(2, dtype=torch.long),
    }]
    indices, qualities = m(outputs, targets, return_quality=True)
    assert len(indices) == 1
    q = qualities[0]
    assert q.shape == (indices[0][1].numel(),)
    assert bool((q >= 0).all() and (q <= 1).all())


# --------------------------------- R2#4: EMA apply_shadow non-float buffers

def test_ema_apply_shadow_leaves_nonfloat_buffers_live():
    from magformer.engine.model_ema import ModelEMA
    torch.manual_seed(7)
    model = torch.nn.Module()
    model.register_buffer("weight_like", torch.randn(3))
    model.register_buffer("step_like", torch.full((1,), 12000,
                                                  dtype=torch.long))
    ema = ModelEMA(model, decay=0.999)
    ema.apply_shadow(model)
    assert int(model.step_like.item()) == 12000, (
        "apply_shadow reset the integer schedule buffer to its init clone")
    assert torch.allclose(
        model.weight_like, ema.shadow["weight_like"]), (
        "float entries must still be swapped in")
    ema.restore(model)
    assert int(model.step_like.item()) == 12000


def test_ema_update_matches_reference_loop():
    from magformer.engine.model_ema import ModelEMA
    torch.manual_seed(8)
    m1 = torch.nn.Linear(4, 4)
    m2 = torch.nn.Linear(4, 4)
    m2.load_state_dict(m1.state_dict())
    e1, e2 = ModelEMA(m1, decay=0.9, warmup_iters=0), ModelEMA(
        m2, decay=0.9, warmup_iters=0)
    for _ in range(3):
        with torch.no_grad():
            m1.weight += 0.1
            m2.weight += 0.1
        e1.update(0, m1)
        e2.update(0, m2)
    for k in e1.shadow:
        torch.testing.assert_close(e1.shadow[k], e2.shadow[k])


# -------------------------------------- R2-SUSPECT: _seed_calls eval gate

def test_seed_calls_do_not_advance_in_eval():
    from magformer.models.common.transformer.multiscale_decoder import (
        MultiScaleMaskedTransformerDecoder)
    from magformer.models.common.layers.position_encoding import (
        PositionEmbeddingSine)
    torch.manual_seed(9)
    dec = MultiScaleMaskedTransformerDecoder(
        num_queries=3, hidden_dim=8, nheads=2, dim_feedforward=16,
        num_layers=1, num_classes=1, mask_dim=8, dropout=0.0,
        num_feature_levels=1, mask_attn_topk_min=2)
    dec._seed_calls = 0
    feat = torch.randn(1, 8, 2, 3)
    padding = torch.zeros(1, 2, 3, dtype=torch.bool)
    pos = PositionEmbeddingSine(4, normalize=True)(feat, padding)
    kwargs = dict(
        memory=feat, mask_features=torch.randn(1, 8, 4, 6),
        multi_scale_pos=[pos], multi_scale_padding_masks=[padding])

    dec.train()
    dec(multi_scale_features=[feat], **kwargs)
    dec(multi_scale_features=[feat], **kwargs)
    assert dec._seed_calls == 2
    dec.eval()
    dec(multi_scale_features=[feat], **kwargs)
    assert dec._seed_calls == 2, "eval forward advanced the seed ramp"
    dec.train()
    dec(multi_scale_features=[feat], **kwargs)
    assert dec._seed_calls == 3


# -------------------------------------- R3: loader snapshot cadence helper

def test_train_snapshot_cadence():
    from tools.train import _train_snapshot_cadence
    cfg = SimpleNamespace(runtime=SimpleNamespace(
        eval_period=8000, checkpoint_period=8000, grad_accum_steps=1))
    assert _train_snapshot_cadence(cfg) == 8000
    cfg = SimpleNamespace(runtime=SimpleNamespace(
        eval_period=200, checkpoint_period=200, grad_accum_steps=4))
    assert _train_snapshot_cadence(cfg) == 800
    cfg = SimpleNamespace(runtime=SimpleNamespace(
        eval_period=0, checkpoint_period=0, grad_accum_steps=1))
    assert _train_snapshot_cadence(cfg) == 1000
    assert _train_snapshot_cadence(SimpleNamespace(runtime=None)) == 1000


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
