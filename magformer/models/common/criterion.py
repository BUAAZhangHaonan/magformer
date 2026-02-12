# -*- coding: utf-8 -*-
"""Set criterion with Mask2Former-compatible semantics."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import HungarianMatcher


def _dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def _sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


def point_sample(input_tensor: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    grid = point_coords * 2.0 - 1.0
    grid = grid.unsqueeze(2)
    sampled = F.grid_sample(input_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled.squeeze(-1)


def calculate_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    return -torch.abs(logits)


def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> torch.Tensor:
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords).squeeze(1)
    uncertainties = calculate_uncertainty(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    idx = torch.topk(uncertainties, k=num_uncertain_points, dim=1)[1]
    idx = idx.unsqueeze(-1).expand(-1, -1, 2)
    uncertain_coords = torch.gather(point_coords, dim=1, index=idx)

    if num_random_points > 0:
        random_coords = torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)
        return torch.cat([uncertain_coords, random_coords], dim=1)
    return uncertain_coords


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float,
        losses: Tuple[str, ...] = ("labels", "masks"),
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = max(float(num_masks), 1.0)

        losses = {}
        for loss_name in self.losses:
            losses.update(self._get_loss(loss_name, outputs, targets, indices, num_masks))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss_name in self.losses:
                    aux_dict = self._get_loss(loss_name, aux_outputs, targets, aux_indices, num_masks)
                    losses.update({f"{k}_{i}": v for k, v in aux_dict.items()})

        total = None
        for key, value in losses.items():
            weight = self.weight_dict.get(key, 1.0)
            weighted = value * weight
            total = weighted if total is None else total + weighted
        losses["total_loss"] = total if total is not None else torch.tensor(0.0, device=outputs["pred_logits"].device)
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _loss_labels(self, outputs, targets, indices, num_masks):
        src_logits = outputs["pred_logits"].float()
        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        if len(indices) > 0 and sum(len(j) for _, j in indices) > 0:
            target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
            target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def _loss_masks(self, outputs, targets, indices, num_masks):
        src_masks = outputs["pred_masks"]
        src_idx = self._get_src_permutation_idx(indices)
        if src_idx[0].numel() == 0:
            loss_mask = torch.tensor(0.0, device=src_masks.device)
            loss_dice = torch.tensor(0.0, device=src_masks.device)
            return {"loss_mask": loss_mask, "loss_dice": loss_dice}

        src_masks = src_masks[src_idx]

        target_masks_list = []
        for b, (_, tgt_ids) in enumerate(indices):
            if tgt_ids.numel() == 0:
                continue
            target_masks_list.append(targets[b]["masks"][tgt_ids].float())

        target_masks = torch.cat(target_masks_list, dim=0).to(src_masks)

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                num_points=self.num_points,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
            )
            point_labels = point_sample(target_masks, point_coords).squeeze(1)

        point_logits = point_sample(src_masks, point_coords).squeeze(1)

        loss_mask = _sigmoid_ce_loss(point_logits, point_labels, num_masks)
        loss_dice = _dice_loss(point_logits, point_labels, num_masks)

        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    def _get_loss(self, loss_name, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self._loss_labels,
            "masks": self._loss_masks,
        }
        return loss_map[loss_name](outputs, targets, indices, num_masks)
