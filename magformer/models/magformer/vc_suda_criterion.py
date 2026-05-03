# -*- coding: utf-8 -*-
"""VC-SUDA Criterion: extends SetCriterion with pseudo-label losses.

Adds quality-weighted pseudo-label loss on top of the standard supervised
SetCriterion losses. Pseudo-labels come from the EMA teacher and are
filtered by quality score — no Hungarian matching needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any

from magformer.models.common.criterion import SetCriterion, _dice_loss, _sigmoid_ce_loss


class VCSUDACriterion(nn.Module):
    """
    Extended criterion for VC-SUDA training.
    
    Composes:
    1. Standard supervised loss (via SetCriterion) on labeled source data
    2. Pseudo-label loss on unlabeled target data (quality-weighted CE + BCE + Dice)
    3. Domain adaptation losses (added externally by trainer)
    
    Pseudo-label loss uses quality-weighted averaging:
        loss = sum(q_i * loss_i) / sum(q_i)
    where q_i is the pseudo-label quality score from PseudoLabelScorer.
    """
    
    def __init__(
        self,
        supervised_criterion: SetCriterion,
        pseudo_weight_ce: float = 2.0,
        pseudo_weight_mask: float = 5.0,
        pseudo_weight_dice: float = 5.0,
    ):
        """
        Args:
            supervised_criterion: Standard SetCriterion for supervised loss
            pseudo_weight_ce: Weight for pseudo-label classification loss
            pseudo_weight_mask: Weight for pseudo-label mask BCE loss
            pseudo_weight_dice: Weight for pseudo-label dice loss
        """
        super().__init__()
        self.supervised_criterion = supervised_criterion
        self.pseudo_weight_ce = pseudo_weight_ce
        self.pseudo_weight_mask = pseudo_weight_mask
        self.pseudo_weight_dice = pseudo_weight_dice
    
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
    
    def pseudo_label_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        pseudo_targets: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute quality-weighted pseudo-label loss.
        
        No Hungarian matching — pseudo_targets are already aligned to student
        query indices by the teacher's predictions.
        
        Args:
            student_outputs: Student model outputs with 'pred_logits', 'pred_masks'
            pseudo_targets: List of dicts, each with:
                'labels': (N,) class labels from teacher
                'masks': (N, H, W) soft masks from teacher (sigmoid'd)
                'quality_scores': (N,) quality scores from PseudoLabelScorer
                
        Returns:
            Loss dict with 'pseudo_loss_ce', 'pseudo_loss_mask', 'pseudo_loss_dice', 'pseudo_total'
        """
        pred_logits = student_outputs["pred_logits"]  # (B, Nq, C)
        pred_masks = student_outputs["pred_masks"]  # (B, Nq, H, W)
        
        B = pred_logits.shape[0]
        device = pred_logits.device
        
        total_ce = torch.tensor(0.0, device=device)
        total_mask = torch.tensor(0.0, device=device)
        total_dice = torch.tensor(0.0, device=device)
        total_quality = torch.tensor(0.0, device=device)
        num_instances = 0
        
        for b in range(B):
            if b >= len(pseudo_targets):
                break
            
            pt = pseudo_targets[b]
            labels = pt.get("labels", None)
            masks = pt.get("masks", None)
            quality_scores = pt.get("quality_scores", None)
            
            if labels is None or masks is None or quality_scores is None:
                continue
            
            if len(labels) == 0:
                continue
            
            N = len(labels)
            num_instances += N
            
            # Student predictions for the same query slots
            # The trainer will have aligned pseudo_targets to student query indices
            # Here we just take the first N query predictions
            s_logits = pred_logits[b, :N]  # (N, C)
            s_masks = pred_masks[b, :N]    # (N, H, W)
            
            # Quality scores as weights
            q = quality_scores.to(device)  # (N,)
            
            # 1. Classification loss: cross-entropy with teacher labels
            # For single-class, labels are all 0 (foreground)
            ce_loss_per_instance = F.cross_entropy(
                s_logits.float(),
                labels.to(device),
                reduction='none'
            )  # (N,)
            total_ce = total_ce + (q * ce_loss_per_instance).sum()
            
            # 2. Mask BCE loss
            # Use teacher soft masks as targets
            mask_bce_per_instance = F.binary_cross_entropy_with_logits(
                s_masks.float().flatten(1),
                masks.to(device).float().flatten(1),
                reduction='none'
            ).mean(dim=1)  # (N,)
            total_mask = total_mask + (q * mask_bce_per_instance).sum()
            
            # 3. Dice loss
            s_masks_sig = s_masks.sigmoid().float().flatten(1)
            t_masks_flat = masks.to(device).float().flatten(1)
            numerator = 2 * (s_masks_sig * t_masks_flat).sum(-1)
            denominator = s_masks_sig.sum(-1) + t_masks_flat.sum(-1)
            dice_per_instance = (1 - (numerator + 1) / (denominator + 1))  # (N,)
            total_dice = total_dice + (q * dice_per_instance).sum()
            
            total_quality = total_quality + q.sum()
        
        # Normalize by total quality weight
        if total_quality > 0:
            norm = total_quality
        else:
            norm = torch.tensor(1.0, device=device)
        
        losses = {
            "pseudo_loss_ce": self.pseudo_weight_ce * total_ce / norm,
            "pseudo_loss_mask": self.pseudo_weight_mask * total_mask / norm,
            "pseudo_loss_dice": self.pseudo_weight_dice * total_dice / norm,
        }
        
        losses["pseudo_total"] = sum(losses.values())
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
