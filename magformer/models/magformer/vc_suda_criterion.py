# -*- coding: utf-8 -*-
"""VC-SUDA Criterion: extends SetCriterion with pseudo-label losses.

Adds quality-weighted pseudo-label loss on top of the standard supervised
SetCriterion losses. Pseudo-labels come from the EMA teacher and are
filtered by quality score. Hungarian matching aligns student queries to
teacher pseudo-labels before loss computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Any, Tuple

from magformer.models.common.criterion import SetCriterion, _dice_loss, _sigmoid_ce_loss


class VCSUDACriterion(nn.Module):
    """
    Extended criterion for VC-SUDA training.

    Composes:
    1. Standard supervised loss (via SetCriterion) on labeled source data
    2. Pseudo-label loss on unlabeled target data (quality-weighted CE + BCE + Dice)
    3. Domain adaptation losses (added externally by trainer)

    Pseudo-label loss uses Hungarian matching to find the optimal assignment
    between student queries and teacher pseudo-labels, then computes
    quality-weighted losses on the matched pairs.
    """

    def __init__(
        self,
        supervised_criterion: SetCriterion,
        pseudo_weight_ce: float = 2.0,
        pseudo_weight_mask: float = 5.0,
        pseudo_weight_dice: float = 5.0,
        pseudo_no_object_weight: float = 0.1,
    ):
        """
        Args:
            supervised_criterion: Standard SetCriterion for supervised loss
            pseudo_weight_ce: Weight for pseudo-label classification loss
            pseudo_weight_mask: Weight for pseudo-label mask BCE loss
            pseudo_weight_dice: Weight for pseudo-label dice loss
            pseudo_no_object_weight: Classification weight for unmatched queries
        """
        super().__init__()
        self.supervised_criterion = supervised_criterion
        self.pseudo_weight_ce = pseudo_weight_ce
        self.pseudo_weight_mask = pseudo_weight_mask
        self.pseudo_weight_dice = pseudo_weight_dice
        self.pseudo_no_object_weight = pseudo_no_object_weight

    def supervised_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute standard supervised loss via SetCriterion.

        Args:
            outputs: Model outputs with 'pred_logits' and 'pred_masks'
            targets: Ground truth targets

        Returns:
            Loss dict from SetCriterion (includes 'total_loss')
        """
        return self.supervised_criterion(outputs, targets)

    @torch.no_grad()
    def _match_pseudo_labels(
        self,
        s_logits: torch.Tensor,
        s_masks: torch.Tensor,
        t_labels: torch.Tensor,
        t_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hungarian-match student queries to teacher pseudo-labels.

        Args:
            s_logits: (Nq, C) student query classification logits.
            s_masks: (Nq, H, W) student query mask logits.
            t_labels: (N,) teacher pseudo-label class indices.
            t_masks: (N, H, W) teacher pseudo-label masks (sigmoid'd).

        Returns:
            src_indices: (N,) indices into student queries (matched).
            tgt_indices: (N,) indices into teacher pseudo-labels (matched).
        """
        Nq = s_logits.shape[0]
        N = t_labels.shape[0]
        device = s_logits.device

        # --- Classification cost: -log_softmax at teacher label positions ---
        # (Nq, N) — lower is better (student predicts teacher's class strongly)
        cost_cls = -F.log_softmax(s_logits.float(), dim=-1)[:, t_labels]

        # --- Mask cost: sigmoid BCE (pairwise, matmul trick for memory efficiency) ---
        # Same approach as HungarianMatcher._bce_cost to avoid O(Nq*N*H*W) expansion.
        s_flat = s_masks.float().flatten(1)          # (Nq, H*W)
        t_flat = t_masks.float().flatten(1)          # (N, H*W)
        hw = s_flat.shape[1]

        pos = F.binary_cross_entropy_with_logits(
            s_flat, torch.ones_like(s_flat), reduction='none')  # (Nq, H*W)
        neg = F.binary_cross_entropy_with_logits(
            s_flat, torch.zeros_like(s_flat), reduction='none')  # (Nq, H*W)
        cost_mask = (torch.mm(pos, t_flat.t()) + torch.mm(neg, (1 - t_flat).t())) / hw
        # cost_mask: (Nq, N)

        # --- Dice cost ---
        s_sig = s_masks.sigmoid().float().flatten(1)   # (Nq, H*W)
        numerator = 2 * torch.mm(s_sig, t_flat.t())    # (Nq, N)
        denominator = s_sig.sum(dim=1, keepdim=True) + t_flat.sum(dim=1, keepdim=True).t()
        cost_dice = 1 - (numerator + 1) / (denominator + 1)  # (Nq, N)

        # --- Combined cost ---
        C = cost_cls + cost_mask + cost_dice

        # Guard against NaN / Inf (e.g. empty masks)
        if not torch.isfinite(C).all():
            C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)

        # Hungarian matching on CPU (matrix is small: Nq x N)
        C_cpu = C.float().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(C_cpu)

        src_indices = torch.as_tensor(row_ind, dtype=torch.int64, device=device)
        tgt_indices = torch.as_tensor(col_ind, dtype=torch.int64, device=device)
        return src_indices, tgt_indices

    def _pseudo_label_loss_for_outputs(
        self,
        student_outputs: Dict[str, torch.Tensor],
        pseudo_targets: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        pred_logits = student_outputs["pred_logits"]  # (B, Nq, C)
        pred_masks = student_outputs["pred_masks"]    # (B, Nq, H, W)

        B = pred_logits.shape[0]
        Nq = pred_logits.shape[1]
        C = pred_logits.shape[2]
        background_class = C - 1
        device = pred_logits.device

        total_ce = torch.tensor(0.0, device=device)
        total_mask = torch.tensor(0.0, device=device)
        total_dice = torch.tensor(0.0, device=device)
        total_ce_weight = torch.tensor(0.0, device=device)
        total_quality = torch.tensor(0.0, device=device)

        for b in range(B):
            if b >= len(pseudo_targets):
                break

            pt = pseudo_targets[b]
            labels = pt.get("labels", None)
            masks = pt.get("masks", None)
            quality_scores = pt.get("quality_scores", None)

            if labels is None or masks is None or quality_scores is None:
                continue

            # Move teacher data to device once
            t_labels = labels.to(device)
            t_masks = masks.to(device).float()
            q_all = quality_scores.to(device).float()

            target_classes = torch.full(
                (Nq,), background_class, dtype=torch.long, device=device
            )
            ce_weights = torch.full(
                (Nq,), float(self.pseudo_no_object_weight), dtype=pred_logits.dtype, device=device
            )

            if len(t_labels) == 0:
                ce_loss_per_query = F.cross_entropy(
                    pred_logits[b].float(), target_classes, reduction='none'
                )
                total_ce = total_ce + (ce_weights * ce_loss_per_query).sum()
                total_ce_weight = total_ce_weight + ce_weights.sum()
                continue

            t_masks_flat = t_masks.flatten(1)

            # Hungarian matching: find best student query for each teacher label
            src_idx, tgt_idx = self._match_pseudo_labels(
                pred_logits[b], pred_masks[b], t_labels, t_masks,
            )

            # Select matched student predictions and reorder teacher targets
            s_logits = pred_logits[b][src_idx]       # (N, C)
            s_masks = pred_masks[b][src_idx]         # (N, H, W)
            s_masks_sig = s_masks.sigmoid().float().flatten(1)  # (N, H*W)
            t_masks_matched = t_masks_flat[tgt_idx]  # (N, H*W)
            t_labels_matched = t_labels[tgt_idx]     # (N,)

            # Quality scores reordered to match tgt_idx
            q = q_all[tgt_idx]   # (N,)

            # 1. Classification loss: matched queries use teacher labels,
            # unmatched queries are constrained to the no-object class.
            target_classes[src_idx] = t_labels_matched
            ce_weights[src_idx] = q.to(ce_weights.dtype)
            ce_loss_per_instance = F.cross_entropy(
                pred_logits[b].float(),
                target_classes,
                reduction='none'
            )  # (Nq,)
            total_ce = total_ce + (ce_weights * ce_loss_per_instance).sum()
            total_ce_weight = total_ce_weight + ce_weights.sum()

            # 2. Mask BCE loss
            mask_bce_per_instance = F.binary_cross_entropy_with_logits(
                s_masks.float().flatten(1),
                t_masks_matched,
                reduction='none'
            ).mean(dim=1)  # (N,)
            total_mask = total_mask + (q * mask_bce_per_instance).sum()

            # 3. Dice loss
            numerator = 2 * (s_masks_sig * t_masks_matched).sum(-1)
            denominator = s_masks_sig.sum(-1) + t_masks_matched.sum(-1)
            dice_per_instance = (1 - (numerator + 1) / (denominator + 1))  # (N,)
            total_dice = total_dice + (q * dice_per_instance).sum()

            total_quality = total_quality + q.sum()

        ce_norm = total_ce_weight.clamp_min(1.0)
        mask_norm = total_quality.clamp_min(1.0)

        losses = {
            "pseudo_loss_ce": self.pseudo_weight_ce * total_ce / ce_norm,
            "pseudo_loss_mask": self.pseudo_weight_mask * total_mask / mask_norm,
            "pseudo_loss_dice": self.pseudo_weight_dice * total_dice / mask_norm,
        }

        losses["pseudo_total"] = sum(losses.values())
        return losses

    def pseudo_label_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        pseudo_targets: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute quality-weighted pseudo-label loss with Hungarian matching.

        Matched queries train against teacher pseudo-labels. Unmatched queries
        train against the no-object/background class. Auxiliary decoder outputs
        receive the same pseudo-label treatment with suffixed loss keys.

        Args:
            student_outputs: Student model outputs with 'pred_logits', 'pred_masks'
            pseudo_targets: List of dicts, each with:
                'labels': (N,) class labels from teacher
                'masks': (N, H, W) soft masks from teacher (sigmoid'd)
                'quality_scores': (N,) quality scores from PseudoLabelScorer

        Returns:
            Loss dict with pseudo classification, mask, dice, and total losses.
        """
        outputs_without_aux = {
            k: v for k, v in student_outputs.items() if k != "aux_outputs"
        }
        losses = self._pseudo_label_loss_for_outputs(outputs_without_aux, pseudo_targets)
        total = losses["pseudo_total"]

        for i, aux_outputs in enumerate(student_outputs.get("aux_outputs", [])):
            aux_losses = self._pseudo_label_loss_for_outputs(aux_outputs, pseudo_targets)
            for key, value in aux_losses.items():
                losses[f"{key}_{i}"] = value
            total = total + aux_losses["pseudo_total"]

        losses["pseudo_total"] = total
        return losses

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, Any]],
        pseudo_targets: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward: supervised + pseudo-label losses.

        Args:
            outputs: Student model outputs
            targets: Ground truth targets (labeled source)
            pseudo_targets: Teacher pseudo-labels (unlabeled target), optional

        Returns:
            Combined loss dict
        """
        # Supervised loss
        losses = self.supervised_loss(outputs, targets)

        # Pseudo-label loss (if available)
        if pseudo_targets is not None:
            pseudo_losses = self.pseudo_label_loss(outputs, pseudo_targets)
            for k, v in pseudo_losses.items():
                losses[k] = v

            # Add pseudo_total to total_loss
            if "pseudo_total" in losses and "total_loss" in losses:
                losses["total_loss"] = losses["total_loss"] + losses["pseudo_total"]

        return losses
