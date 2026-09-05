from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from magformer.config.schema import MaskFormerConfig
from magformer.models.common.box_ops import masks_to_boxes_cxcywh
from magformer.models.common.criterion import SetCriterion
from magformer.models.common.matcher import HungarianMatcher
from magformer.models.magformer.arch import MagFormerArch


def _target_mask() -> torch.Tensor:
    mask = torch.zeros((1, 4, 4), dtype=torch.float32)
    mask[0, 1:3, 2:4] = 1.0
    return mask


def test_masks_to_boxes_uses_geometric_center_and_exclusive_edges() -> None:
    masks = torch.cat((_target_mask(), torch.zeros((1, 4, 4))), dim=0)
    masks[1, 0, 0] = 1.0

    boxes = masks_to_boxes_cxcywh(masks)

    torch.testing.assert_close(
        boxes,
        torch.tensor(
            [
                [0.75, 0.50, 0.50, 0.50],
                [0.125, 0.125, 0.25, 0.25],
            ]
        ),
    )


def test_exact_box_prediction_has_zero_l1_and_giou_losses() -> None:
    target_masks = _target_mask()
    exact_box = masks_to_boxes_cxcywh(target_masks)
    criterion = SetCriterion(
        num_classes=1,
        matcher=nn.Identity(),
        weight_dict={},
        eos_coef=0.1,
        losses=("boxes",),
    )
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    losses = criterion._loss_boxes(
        {"pred_boxes": exact_box.unsqueeze(0)},
        [{"labels": torch.tensor([0]), "masks": target_masks}],
        indices,
        num_masks=1.0,
    )

    torch.testing.assert_close(losses["loss_bbox"], torch.tensor(0.0), atol=1e-7, rtol=0.0)
    torch.testing.assert_close(losses["loss_giou"], torch.tensor(0.0), atol=1e-7, rtol=0.0)


def test_box_costs_select_the_exact_cxcywh_query() -> None:
    target_masks = _target_mask()
    exact_box = masks_to_boxes_cxcywh(target_masks)[0]
    matcher = HungarianMatcher(
        cost_class=0.0,
        cost_mask=0.0,
        cost_dice=0.0,
        num_points=16,
        cost_bbox=5.0,
        cost_giou=2.0,
    )
    outputs = {
        "pred_logits": torch.zeros((1, 2, 1)),
        "pred_masks": torch.zeros((1, 2, 4, 4)),
        "pred_boxes": torch.stack(
            (torch.tensor([0.125, 0.125, 0.25, 0.25]), exact_box)
        ).unsqueeze(0),
    }

    indices = matcher(
        outputs,
        [{"labels": torch.tensor([0]), "masks": target_masks}],
    )

    torch.testing.assert_close(indices[0][0], torch.tensor([1]))
    torch.testing.assert_close(indices[0][1], torch.tensor([0]))


def test_box_cost_config_is_typed_non_negative_and_wired() -> None:
    mask_former = MaskFormerConfig(cost_bbox=5.0, cost_giou=2.0)
    model = MagFormerArch.__new__(MagFormerArch)
    nn.Module.__init__(model)
    model.num_classes = 1

    model._sync_criterion_from_config(SimpleNamespace(mask_former=mask_former))

    assert model.criterion.matcher.cost_bbox == 5.0
    assert model.criterion.matcher.cost_giou == 2.0
    with pytest.raises(ValidationError):
        MaskFormerConfig(cost_bbox=-1.0)
    with pytest.raises(ValidationError):
        MaskFormerConfig(cost_giou=-1.0)
