"""Final-review fix regressions (2026-09-21, reviewers #1/#2 findings):

1. DN positive layout: flat slot i must decode GT-major (g = i // dn_scalar)
   to match the loss-side decode; was scalar-major (scrambled pairing).
2. DN small-GT window frame: bbox extraction runs in the GT mask's own
   (1024^2) frame; was clamped/normalized by the 512^2 prediction grid
   (truncated/inverted windows, coords > 1).
3. HDA+ hires anneal actually executes via build_warmup_cosine_scheduler;
   main groups stay bit-identical.
"""

from types import SimpleNamespace

import torch

from magformer.engine.lr_scheduler import build_warmup_cosine_scheduler
from magformer.models.magformer.arch import MagFormerArch


# ---------------------------------------------------------------- test 1
def _dn_stub(cap=8, scalar=3, hidden=32, num_queries=64, noise=0.0):
    return SimpleNamespace(
        dn_enabled=True, training=True,
        dn_scalar=scalar, dn_box_noise_scale=noise, dn_max_gt_cap=cap,
        num_classes=1,
        query_embed=SimpleNamespace(embedding_dim=hidden,
                                    weight=torch.nn.Parameter(torch.randn(num_queries, hidden))),
        _boxes_from_masks=MagFormerArch._boxes_from_masks,
        _sinusoidal_box_pe=MagFormerArch._sinusoidal_box_pe,
    )


def _two_gt_masks(H=1024, W=1024):
    masks = torch.zeros(2, H, W)
    masks[0, 400:500, 400:520] = 1.0   # distinct boxes
    masks[1, 60:120, 700:760] = 1.0
    return masks


def test_dn_positive_layout_is_gt_major():
    """With zero noise, slot i's PE must equal PE(gt_box[i // dn_scalar])."""
    torch.manual_seed(0)
    stub = _dn_stub()
    masks = _two_gt_masks()
    labels = torch.zeros(1, 2, dtype=torch.long)
    pad = torch.ones(1, 2, dtype=torch.bool)  # True = valid slot
    emb, feat, meta = MagFormerArch._generate_dn_queries(
        stub, None, None, labels, masks[None], pad)
    scalar, cap = stub.dn_scalar, stub.dn_max_gt_cap
    num_pos = scalar * cap
    assert emb.shape == (cap * (scalar + 1), 1, 32)  # pos rows first, then neg
    gt_boxes = MagFormerArch._boxes_from_masks(masks)  # (2, 4) cxcywh
    nv = 2
    for i in range(nv * scalar):  # valid prefix; g >= nv rows are zero-box PEs
        g = i // scalar
        pe = MagFormerArch._sinusoidal_box_pe(gt_boxes[g][None, None], 32)
        assert torch.allclose(emb[i, 0], pe[0, 0], atol=1e-5), (
            f"slot {i}: PE does not belong to its GT-major GT {g}")


# ---------------------------------------------------------------- test 2
def test_dn_small_gt_window_uses_gt_frame(monkeypatch):
    """Window coords for a GT at rows 600-700 must stay in [0,1] and center
    on the GT's relative bbox (the 512-grid frame produced coords ~2x)."""
    import magformer.models.common.criterion as crit
    from magformer.models.common.criterion import SetCriterion

    captured = []
    orig_point_sample = crit.point_sample

    def spy_point_sample(tensor, coords, **kw):
        if coords.dim() == 3 and coords.shape[-1] == 2 and tensor.dim() == 4:
            captured.append(coords.detach().clone())
        return orig_point_sample(tensor, coords, **kw)

    monkeypatch.setattr(crit, "point_sample", spy_point_sample)

    crit_obj = SetCriterion(
        num_classes=1, matcher=None, weight_dict={}, eos_coef=0.1,
        losses=("labels",), num_points=64, dn_enabled=True,
    )
    crit_obj.dn_loss_weight = 1.0
    crit_obj.dn_contrastive_weight = 0.0

    H = W = 1024
    gt = torch.zeros(1, H, W)
    gt[0, 600:650, 600:650] = 1.0   # 2500px^2 < 4096: window branch; below/right of 511
    scalar, cap = 3, 8
    num_pos, num_dn = scalar * cap, cap * (scalar + 1)
    outputs = {
        "pred_logits": torch.randn(1, num_dn + 10, 2),
        "pred_masks": torch.rand(1, num_dn + 10, 512, 512),
        "dn_enabled": True,
        "dn_num_regular_queries": 10,
        "dn_meta": {
            "num_positives": num_pos, "num_negatives": cap,
            "dn_scalar": scalar, "max_valid": cap,
            "batch_num_valid": [1],
            "gt_labels_padded": torch.zeros(1, cap, dtype=torch.long),
            "neg_valid": torch.zeros(1, cap, dtype=torch.bool),
        },
    }
    crit_obj._compute_dn_loss(outputs, [{"labels": torch.tensor([0]), "masks": gt}], outputs["dn_meta"])

    assert captured, "no window coords captured"
    for coords in captured:
        assert (coords >= -1e-6).all() and (coords <= 1.0 + 1e-6).all(), (
            f"window coords escaped [0,1]: [{coords.min():.3f}, {coords.max():.3f}]")
    ys = torch.cat([c[..., 1].flatten() for c in captured])
    rel_center = 625.0 / (H - 1)
    assert abs(float(ys.mean()) - rel_center) < 0.15, (
        f"window y-centre {float(ys.mean()):.3f} far from GT centre {rel_center:.3f}")


# ---------------------------------------------------------------- test 3
def test_hires_anneal_executes_main_bit_identical():
    def make(tagged):
        g1 = torch.nn.Linear(4, 4)
        g2 = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW([
            {"params": g1.parameters(), "lr": 1e-5},
            {"params": g2.parameters(), "lr": 2e-4, **({"hires_mult": 20.0} if tagged else {})},
        ])
        return opt

    max_iter, warmup = 256000, 2500
    plain = build_warmup_cosine_scheduler(make(False), max_iter, warmup, 0.001)
    anneal = build_warmup_cosine_scheduler(make(True), max_iter, warmup, 0.001,
                                           hires_mult_anneal=True)
    t0, t1 = int(0.1 * max_iter), int(0.5 * max_iter)
    for it in (0, warmup, t0 // 2, t0, (t0 + t1) // 2, t1, max_iter - 1):
        # main group (index 0): bit-identical schedules
        assert plain.get_last_lr() is not None
        for sched in (plain, anneal):
            sched.step = None if False else None
        # compare lambdas directly (lr = base_lr * lambda)
        lam_plain = plain.lr_lambdas[0](it)
        lam_anneal_main = anneal.lr_lambdas[0](it)
        assert lam_plain == lam_anneal_main, f"main group diverged at {it}"
        lam_hires = anneal.lr_lambdas[1](it)
        if it <= t0:
            eff = lam_hires / lam_anneal_main
            assert abs(eff - 1.0) < 1e-12, f"pre-anneal hires factor {eff}"
        if it >= t1:
            eff = lam_hires / lam_anneal_main
            assert abs(eff - 1.0 / 20.0) < 1e-9, f"post-anneal hires factor {eff}"
    mid = (t0 + t1) // 2
    eff_mid = anneal.lr_lambdas[1](mid) / anneal.lr_lambdas[0](mid)
    assert 0.2 < eff_mid < 0.9, f"mid-anneal factor {eff_mid} not between 1.0 and 1/20"
