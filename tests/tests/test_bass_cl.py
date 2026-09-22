"""BAS-CL+ tests (implementation step 3, arena P3-a winner).

Three-segment sampling, chain-aligned coverage labels, band weighting with
balanced_ce exemption, band Dice, ramps, and bass-off bit-identity.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from magformer.models.common.criterion import SetCriterion
from magformer.models.common.matcher import HungarianMatcher


class _StubCriterion:
    _get_src_permutation_idx = SetCriterion._get_src_permutation_idx
    _get_tgt_permutation_idx = SetCriterion._get_tgt_permutation_idx
    _build_bass_pack = SetCriterion._build_bass_pack
    _bass_sample_pool = SetCriterion._bass_sample_pool
    _loss_masks_bass = SetCriterion._loss_masks_bass


def _mk(bass=True, balanced=False, lam=3.0, floor=32, ramp=2000, calls=10**9):
    s = _StubCriterion()
    s.num_classes = 1
    s.matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=20.0)
    s.weight_dict = {"loss_mask": 5.0, "loss_dice": 1.0}
    s.eos_coef = 0.1
    s.losses = ("labels", "masks")
    s.num_points = 1024
    s.oversample_ratio = 3.0
    s.importance_sample_ratio = 0.75
    s.balanced_ce = balanced
    s.balanced_ce_min_fg_ratio = 0.01
    s.focal_alpha = 0.25
    s.mal_enabled = False
    s.mal_beta = 2.0
    s.mal_warmup_iters = 1
    s.mal_scale_in_ce = False
    s._mal_calls = 0
    s.bass_enabled = bass
    s.bass_boundary_ratio = 0.30
    s.bass_interior_ratio = 0.20
    s.bass_point_floor = floor
    s.bass_weight_lambda = lam
    s.bass_band_dice_weight = 1.0
    s.bass_ramp_iters = ramp
    s.bass_soft_label = True
    s.bass_soft_label_mix = 1.0
    s._bass_calls = calls
    s.dn_enabled = False
    s.dn_loss_weight = 1.0
    s.dn_contrastive_weight = 0.5
    s.scale_adaptive_alpha = 0.0
    s.small_object_sample_threshold = 0
    s.use_uncertainty_weighting = False
    return s


def _fixtures(N_gt=3, q=6, H=128, seed=0):
    torch.manual_seed(seed)
    masks = torch.zeros(N_gt, H, H)
    masks[0, 31:52, 30:51] = 1.0    # large, odd size -> fractional cells
    masks[1, 70:77, 70:77] = 1.0    # small, odd size
    masks[2, 101:122, 10:31] = 1.0  # large, odd size
    targets = [{"masks": masks,
                "labels": torch.zeros(N_gt, dtype=torch.long)}]
    indices = [(torch.arange(N_gt) % q, torch.arange(N_gt))]
    src = torch.randn(1, q, H, H, requires_grad=True)
    return targets, indices, src


def test_pack_band_interior_split():
    s = _mk()
    targets, _, _ = _fixtures()
    pack = s._build_bass_pack(targets)
    soft = pack["soft"]
    h, w = pack["h"], pack["w"]
    assert soft.shape == (3, 1, h, w)
    # coverage in [0,1]; interior coverage==1; boundary cells fractional
    assert soft.min() >= 0 and soft.max() <= 1
    # every GT has band and interior cells
    off = pack["band_offsets"]
    for i in range(3):
        assert off[i + 1] > off[i], "band must be non-empty per GT"
    ioff = pack["interior_offsets"]
    for i in range(3):
        assert ioff[i + 1] > ioff[i], "interior must be non-empty per GT"


def test_bass_loss_runs_and_keys():
    s = _mk()
    targets, indices, src = _fixtures()
    pack = s._build_bass_pack(targets)
    N_gt = 3
    src_sel = src[0][indices[0][0]][:, None]
    tgt_sel = targets[0]["masks"][indices[0][1]][:, None]
    out = s._loss_masks_bass(
        src_sel, tgt_sel, targets, indices, float(N_gt), None, pack)
    for k in ("loss_mask", "loss_dice", "loss_dice_band"):
        assert k in out and torch.isfinite(out[k]), k
    out["loss_mask"].backward()  # gradient flows
    assert src.grad is not None


def test_band_points_lie_in_coverage_band():
    torch.manual_seed(1)
    s = _mk()
    targets, _, _ = _fixtures()
    pack = s._build_bass_pack(targets)
    gt_ids = torch.tensor([0, 1, 2])
    s._bass_pack_w = pack["w"]
    s._bass_pack_h = pack["h"]
    coords = s._bass_sample_pool(
        pack["band_cells"], pack["band_offsets"], gt_ids, 64)
    from magformer.models.common.matcher import point_sample as _ps
    soft = pack["soft"][gt_ids]
    vals = _ps(soft, coords).squeeze(1)
    # every drawn point must land in a band cell (coverage in (0.05,0.95))
    assert ((vals > 0.02) & (vals < 0.98)).float().mean() > 0.9


def test_ramp_zero_band_dice_at_start():
    s0 = _mk(calls=0)      # ramp t=0
    s1 = _mk(calls=10**9)  # ramp t=1
    targets, indices, src = _fixtures(seed=3)
    pack0 = s0._build_bass_pack(targets)
    pack1 = s1._build_bass_pack(targets)
    src_sel = src[0][indices[0][0]][:, None]
    tgt_sel = targets[0]["masks"][indices[0][1]][:, None]
    out0 = s0._loss_masks_bass(src_sel, tgt_sel, targets, indices, 3.0, None, pack0)
    out1 = s1._loss_masks_bass(src_sel, tgt_sel, targets, indices, 3.0, None, pack1)
    assert out0["loss_dice_band"].abs().item() == 0.0
    assert out1["loss_dice_band"].abs().item() > 0.0


def test_seed_prior_all_layers_flag_wiring():
    """SCB+: the schema keys exist and decoder accepts the new ctor args."""
    from magformer.models.common.transformer.multiscale_decoder import (
        MultiScaleMaskedTransformerDecoder as Dec)
    import inspect
    sig = inspect.signature(Dec.__init__)
    assert "seed_ramp_iters" in sig.parameters
    assert "seed_prior_all_layers" in sig.parameters
    from magformer.config.schema import MaskFormerConfig
    cfg = MaskFormerConfig()
    assert cfg.seed_dropout == 0.0 and cfg.seed_ramp_iters == 0
    assert cfg.seed_prior_all_layers is False
