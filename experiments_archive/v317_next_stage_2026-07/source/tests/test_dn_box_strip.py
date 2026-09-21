"""Regression: DN query rows must be stripped from pred_boxes, not just
pred_logits/pred_masks, before matching.

With dn_enabled + hybrid box costs (cost_bbox/cost_giou > 0) the decoder's
heads emit pred_boxes over the FULL query stream (regular + DN rows); the
criterion's DN strip used to leave pred_boxes unsliced so the matcher mixed
200-row class/mask costs with 232-row box costs and crashed at
matcher.py:435 (found by the F2 merge smoke, 2026-09-21; would have killed
the G1 B1 arm at step 1).
"""

from typing import Any, Dict

import torch

from magformer.models.common.criterion import SetCriterion
from magformer.models.common.matcher import HungarianMatcher

NUM_REGULAR = 6
NUM_DN = 4
Q_TOTAL = NUM_REGULAR + NUM_DN


def _criterion() -> SetCriterion:
    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_mask=5.0,
        cost_dice=20.0 / 20.0,
        num_points=16,
        cost_bbox=5.0,
        cost_giou=2.0,
    )
    crit = SetCriterion(
        num_classes=1,
        matcher=matcher,
        weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 1.0,
                     "loss_bbox": 5.0, "loss_giou": 2.0},
        eos_coef=0.1,
        losses=("labels", "masks", "boxes"),
        num_points=16,
        dn_enabled=True,
    )
    # The DN-loss internals have their own tests; this test targets the strip.
    crit._compute_dn_loss = lambda *a, **k: {}  # type: ignore[method-assign]
    return crit


def _outputs(with_aux: bool) -> Dict[str, Any]:
    torch.manual_seed(7)
    H = W = 16
    out = {
        # regular rows carry a learnable signal so matching is non-trivial
        "pred_logits": torch.randn(1, Q_TOTAL, 1) * 2.0,
        "pred_masks": torch.sigmoid(torch.randn(1, Q_TOTAL, H, W)),
        "pred_boxes": torch.rand(1, Q_TOTAL, 4) * 0.4 + 0.3,
        "dn_enabled": True,
        "dn_num_regular_queries": NUM_REGULAR,
        "dn_meta": {"stub": True},
    }
    if with_aux:
        out["aux_outputs"] = [dict(out) for _ in range(2)]
        for aux in out["aux_outputs"]:
            aux.pop("aux_outputs", None)
            aux.pop("dn_meta", None)
    return out


def _targets() -> list:
    m1 = torch.zeros(16, 16)
    m1[4:10, 6:12] = 1.0
    m2 = torch.zeros(16, 16)
    m2[2:5, 2:5] = 1.0
    return [
        {"labels": torch.tensor([0, 0]), "masks": torch.stack((m1, m2))},
    ]


def test_forward_with_dn_and_box_costs_runs() -> None:
    crit = _criterion()
    losses = crit(_outputs(with_aux=False), _targets())
    assert torch.isfinite(losses["total_loss"])
    assert "loss_bbox" in losses and "loss_giou" in losses


def test_forward_with_dn_box_costs_and_aux_runs() -> None:
    crit = _criterion()
    losses = crit(_outputs(with_aux=True), _targets())
    assert torch.isfinite(losses["total_loss"])
    assert "loss_bbox_0" in losses and "loss_giou_1" in losses


def test_dn_rows_never_reach_matcher() -> None:
    """The matcher must see exactly NUM_REGULAR rows in every cost input."""
    seen = {}

    class SpyMatcher(HungarianMatcher):
        def forward(self, outputs, targets, return_quality: bool = False):
            seen["logits"] = int(outputs["pred_logits"].shape[1])
            seen["masks"] = int(outputs["pred_masks"].shape[1])
            seen["boxes"] = int(outputs["pred_boxes"].shape[1])
            return super().forward(outputs, targets, return_quality=return_quality)

    crit = _criterion()
    crit.matcher = SpyMatcher(
        cost_class=2.0, cost_mask=5.0, cost_dice=1.0, num_points=16,
        cost_bbox=5.0, cost_giou=2.0)
    crit(_outputs(with_aux=True), _targets())
    assert seen["logits"] == NUM_REGULAR
    assert seen["masks"] == NUM_REGULAR
    assert seen["boxes"] == NUM_REGULAR
