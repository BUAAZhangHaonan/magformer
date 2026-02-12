# -*- coding: utf-8 -*-
"""
Set Criterion

Minimal loss computation for single-class MaskFormer-style training.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import HungarianMatcher


def _dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denom = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denom + 1)
    return loss


def _sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")


class SetCriterion(nn.Module):
    def __init__(
        self,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)
        loss_dict = {}

        loss_dict.update(self._loss_labels(outputs, targets, indices))
        loss_dict.update(self._loss_masks(outputs, targets, indices))

        total = None
        for key, value in loss_dict.items():
            weight = self.weight_dict.get(key, 1.0)
            total = value * weight if total is None else total + value * weight
        loss_dict["total_loss"] = total if total is not None else torch.tensor(0.0, device=outputs["pred_logits"].device)
        return loss_dict

    def _loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"].squeeze(-1)
        losses = []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src = src_logits[b, src_idx]
            tgt = torch.ones_like(src)
            losses.append(F.binary_cross_entropy_with_logits(src, tgt))

        if len(losses) == 0:
            loss = torch.tensor(0.0, device=src_logits.device)
        else:
            loss = torch.stack(losses).mean()
        return {"loss_ce": loss}

    def _loss_masks(self, outputs, targets, indices):
        src_masks = outputs["pred_masks"]
        losses_mask = []
        losses_dice = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src = src_masks[b, src_idx]
            tgt = targets[b]["masks"].float()[tgt_idx]

            if tgt.shape[-2:] != src.shape[-2:]:
                tgt = F.interpolate(tgt[:, None], size=src.shape[-2:], mode="nearest")[:, 0]

            src_flat = src.flatten(1)
            tgt_flat = tgt.flatten(1)

            losses_mask.append(_sigmoid_ce_loss(src_flat, tgt_flat).mean())
            losses_dice.append(_dice_loss(src_flat, tgt_flat).mean())

        if len(losses_mask) == 0:
            loss_mask = torch.tensor(0.0, device=src_masks.device)
            loss_dice = torch.tensor(0.0, device=src_masks.device)
        else:
            loss_mask = torch.stack(losses_mask).mean()
            loss_dice = torch.stack(losses_dice).mean()

        return {"loss_mask": loss_mask, "loss_dice": loss_dice}
