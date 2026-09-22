from __future__ import annotations

import torch
from torch import nn

from magformer.models.common.criterion import SetCriterion


def _loss_and_grad(eos_coef: float) -> tuple[torch.Tensor, torch.Tensor]:
    criterion = SetCriterion(
        num_classes=1,
        matcher=nn.Identity(),
        weight_dict={},
        eos_coef=eos_coef,
        losses=("labels",),
    )
    logits = torch.zeros((1, 2, 2), dtype=torch.float32, requires_grad=True)
    targets = [{"labels": torch.tensor([0], dtype=torch.int64)}]
    indices = [
        (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )
    ]
    loss = criterion._loss_labels(
        {"pred_logits": logits},
        targets,
        indices,
        num_masks=1.0,
    )["loss_ce"]
    loss.backward()
    return loss.detach(), logits.grad.detach().clone()


def test_no_object_weight_scales_only_unmatched_query_loss_and_gradient() -> None:
    full_loss, full_grad = _loss_and_grad(1.0)
    weighted_loss, weighted_grad = _loss_and_grad(0.1)

    assert weighted_loss < full_loss
    torch.testing.assert_close(weighted_grad[:, 0], full_grad[:, 0])
    torch.testing.assert_close(weighted_grad[:, 1], full_grad[:, 1] * 0.1)
