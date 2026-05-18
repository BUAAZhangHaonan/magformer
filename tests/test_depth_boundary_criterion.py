from __future__ import annotations

import pytest
import torch

from magformer.models.common.criterion import SetCriterion
from magformer.models.magformer.arch import MagFormerArch


class _FixedMatcher:
    def __init__(self, src: torch.Tensor | None = None, tgt: torch.Tensor | None = None):
        self.src = src if src is not None else torch.tensor([0])
        self.tgt = tgt if tgt is not None else torch.tensor([0])

    def __call__(self, outputs, targets):
        del outputs
        return [
            (
                self.src.to(target["labels"].device),
                self.tgt.to(target["labels"].device),
            )
            for target in targets
        ]


def _criterion(*, losses=("depth_boundary",), depth_boundary_weight: float = 1.0):
    return SetCriterion(
        num_classes=1,
        matcher=_FixedMatcher(),
        weight_dict={
            "loss_ce": 1.0,
            "loss_mask": 0.0,
            "loss_dice": 0.0,
            "loss_depth_boundary": depth_boundary_weight,
        },
        eos_coef=0.1,
        losses=losses,
        num_points=8,
        contrastive_enabled=False,
    )


def _outputs():
    pred_masks = torch.randn(1, 2, 4, 4, requires_grad=True)
    return {
        "pred_logits": torch.tensor([[[4.0, -4.0], [-4.0, 4.0]]]),
        "pred_masks": pred_masks,
    }


def _target_with_depth():
    depth = torch.tensor(
        [[
            [0.0, 0.2, 0.4, 0.4, 0.4, 0.4],
            [0.0, 0.2, 0.4, 0.9, 0.9, 0.9],
            [0.0, 0.2, 0.4, 0.9, 0.9, 0.9],
            [0.0, 0.2, 0.4, 0.9, 0.9, 0.9],
            [0.0, 0.2, 0.4, 0.9, 0.9, 0.9],
            [0.0, 0.2, 0.4, 0.9, 0.9, 0.9],
        ]],
        dtype=torch.float32,
    )
    return {
        "labels": torch.tensor([0]),
        "masks": torch.ones(1, 6, 6),
        "depth": depth,
    }


def test_depth_boundary_loss_is_finite_and_backprops_to_pred_masks():
    criterion = _criterion()
    outputs = _outputs()

    losses = criterion(outputs, [_target_with_depth()])

    assert "loss_depth_boundary" in losses
    assert torch.isfinite(losses["loss_depth_boundary"])
    losses["loss_depth_boundary"].backward()
    assert outputs["pred_masks"].grad is not None
    assert outputs["pred_masks"].grad.abs().sum() > 0


def test_depth_boundary_loss_requires_depth_when_enabled():
    criterion = _criterion()
    outputs = _outputs()
    targets = [{"labels": torch.tensor([0]), "masks": torch.ones(1, 6, 6)}]

    with pytest.raises(ValueError, match="loss_depth_boundary requires target depth"):
        criterion(outputs, targets)


def test_zero_weight_depth_boundary_loss_is_omitted_and_does_not_require_depth():
    criterion = _criterion(
        losses=("labels", "depth_boundary"),
        depth_boundary_weight=0.0,
    )
    outputs = _outputs()
    targets = [{"labels": torch.tensor([0]), "masks": torch.ones(1, 6, 6)}]

    losses = criterion(outputs, targets)

    assert "loss_ce" in losses
    assert "loss_depth_boundary" not in losses
    torch.testing.assert_close(losses["total_loss"], losses["loss_ce"])


def test_prepare_targets_preserves_depths_by_batch_index():
    model = object.__new__(MagFormerArch)
    torch.nn.Module.__init__(model)
    model.num_classes = 1
    model.register_buffer("pixel_mean", torch.zeros(3, 1, 1), False)
    depths = torch.arange(2 * 1 * 5 * 7, dtype=torch.float32).view(2, 1, 5, 7)
    targets = [
        {"labels": torch.tensor([1]), "masks": torch.ones(1, 5, 7)},
        {"labels": torch.tensor([1]), "masks": torch.zeros(1, 5, 7)},
    ]

    prepared = model._prepare_targets(targets, depths=depths)

    assert [tuple(target["depth"].shape) for target in prepared] == [(1, 5, 7), (1, 5, 7)]
    torch.testing.assert_close(prepared[0]["depth"], depths[0])
    torch.testing.assert_close(prepared[1]["depth"], depths[1])
