# -*- coding: utf-8 -*-
"""Domain Adaptation Losses for VC-SUDA.

Three domain alignment losses to bridge sim2real gap:
1. PrototypeAlignmentLoss: Align per-class feature prototypes across domains
2. BoundaryConsistencyLoss: Enforce depth-boundary alignment (one-sided)
3. ModalityDropoutConsistencyLoss: RGBD vs RGB prediction consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PrototypeAlignmentLoss(nn.Module):
    """
    Align per-class feature prototypes between source and target domains.
    
    Maintains EMA buffers for source/target prototypes (one per class).
    Loss = L2 distance between matched prototypes across domains.
    
    Since num_classes=1, we maintain a single foreground prototype per domain.
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        feature_dim: int = 256,
        ema_rate: float = 0.9,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.ema_rate = ema_rate
        
        # Prototype buffers: one vector per class per domain
        # Source prototypes (updated from labeled source features)
        self.register_buffer(
            "source_prototypes",
            torch.zeros(num_classes, feature_dim)
        )
        # Target prototypes (updated from student features on target data)
        self.register_buffer(
            "target_prototypes",
            torch.zeros(num_classes, feature_dim)
        )
        self.register_buffer('_source_initialized', torch.tensor(False))
        self.register_buffer('_target_initialized', torch.tensor(False))
    
    @torch.no_grad()
    def _update_prototypes(
        self,
        prototype: torch.Tensor,
        domain: str,             # "source" or "target"
    ):
        """Update domain prototypes via EMA."""
        new_proto = prototype.detach()

        if domain == "source":
            if not self._source_initialized.item():
                self.source_prototypes[0].copy_(new_proto)
                self._source_initialized.fill_(True)
            else:
                self.source_prototypes[0].mul_(self.ema_rate).add_(
                    new_proto, alpha=1 - self.ema_rate
                )
        else:
            if not self._target_initialized.item():
                self.target_prototypes[0].copy_(new_proto)
                self._target_initialized.fill_(True)
            else:
                self.target_prototypes[0].mul_(self.ema_rate).add_(
                    new_proto, alpha=1 - self.ema_rate
                )

    def _compute_soft_prototype(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Pool foreground features with differentiable soft masks."""
        B, D, H, W = features.shape

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)  # (B, 1, H, W)

        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(
                masks.float(), size=(H, W), mode="bilinear", align_corners=False
            )

        masks = masks.float().clamp(0.0, 1.0)
        soft_union = 1.0 - torch.prod(1.0 - masks, dim=1)  # (B, H, W)
        weights = soft_union.flatten(1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (features.flatten(2) * weights.unsqueeze(1)).sum(dim=2) / denom
        return pooled.mean(dim=0)  # (D,)
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_masks: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute prototype alignment loss.
        
        Args:
            source_features: (B, D, H, W) pixel features from source images
            target_features: (B, D, H, W) pixel features from target images
            source_masks: (B, N, H, W) predicted masks on source (sigmoid output)
            target_masks: (B, N, H, W) predicted masks on target (sigmoid output)
            
        Returns:
            Scalar L2 loss between source and target prototypes
        """
        source_proto = self._compute_soft_prototype(source_features, source_masks)
        target_proto = self._compute_soft_prototype(target_features, target_masks)

        self._update_prototypes(source_proto, "source")
        self._update_prototypes(target_proto, "target")

        loss = F.mse_loss(source_proto, target_proto)
        return loss


class BoundaryConsistencyLoss(nn.Module):
    """
    One-sided boundary consistency: enforce that depth gradients align with
    mask boundaries, but only where source has clear boundaries.
    
    This prevents the model from hallucinating boundaries in target domain
    regions where depth is smooth.
    
    Loss: BCE between mask boundary and depth gradient, gated by depth sharpness.
    """
    
    def __init__(
        self,
        boundary_confidence_threshold: float = 0.5,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.threshold = boundary_confidence_threshold
        self.kernel_size = kernel_size
        self._sobel_x = None
        self._sobel_y = None
    
    def _get_sobel_kernels(self, device: torch.device) -> tuple:
        if self._sobel_x is None or self._sobel_x.device != device:
            self._sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32, device=device
            ).reshape(1, 1, 3, 3)
            self._sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=torch.float32, device=device
            ).reshape(1, 1, 3, 3)
        return self._sobel_x, self._sobel_y
    
    def _compute_mask_boundary(self, masks: torch.Tensor) -> torch.Tensor:
        """Compute a differentiable soft boundary from mask probabilities."""
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)  # (B, 1, H, W)
        sobel_x, sobel_y = self._get_sobel_kernels(masks.device)
        grad_x = F.conv2d(masks.float(), sobel_x, padding=1)
        grad_y = F.conv2d(masks.float(), sobel_y, padding=1)
        boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8).squeeze(1)
        max_per_image = boundary.flatten(1).amax(dim=1).view(-1, 1, 1)
        return boundary / (max_per_image + 1e-6)
    
    def _compute_depth_gradient(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute depth gradient magnitude via Sobel."""
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)  # (B, 1, H, W)
        sobel_x, sobel_y = self._get_sobel_kernels(depth.device)
        grad_x = F.conv2d(depth, sobel_x, padding=1)
        grad_y = F.conv2d(depth, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return grad_mag.squeeze(1)  # (B, H, W)
    
    def forward(
        self,
        pred_masks: torch.Tensor,
        depth_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute one-sided boundary consistency loss.
        
        Args:
            pred_masks: (B, N, H, W) predicted masks (after sigmoid)
            depth_maps: (B, 1, H, W) or (B, H, W) depth maps
            
        Returns:
            Scalar boundary consistency loss
        """
        # Aggregate masks: differentiable union of all instances per image
        B = pred_masks.shape[0]
        if pred_masks.ndim == 4:
            probs = pred_masks.float().clamp(0.0, 1.0)
            mask_union = 1.0 - torch.prod(1.0 - probs, dim=1)  # (B, H, W)
        else:
            mask_union = pred_masks.float().clamp(0.0, 1.0)
        
        # Mask boundary
        mask_boundary = self._compute_mask_boundary(mask_union)  # (B, H, W)
        
        # Depth gradient
        depth_grad = self._compute_depth_gradient(depth_maps)  # (B, H, W)
        
        # Normalize depth gradient to [0, 1] for BCE
        depth_grad_norm = depth_grad / (depth_grad.flatten(1).max(dim=1, keepdim=True).values.unsqueeze(1) + 1e-6)
        
        # Gate: only penalize where depth gradient is significant
        gate = (depth_grad_norm > self.threshold).float()
        
        if gate.sum() == 0:
            return pred_masks.sum() * 0.0
        
        loss = F.mse_loss(
            mask_boundary * gate,
            depth_grad_norm.detach() * gate,
            reduction='sum'
        ) / (gate.sum() + 1e-6)
        
        return loss


class ModalityDropoutConsistencyLoss(nn.Module):
    """
    Consistency between full-modality (RGBD) and dropped-modality (RGB only) predictions.
    
    With probability p, zero out depth and forward the student model.
    Enforce MSE between RGBD and RGB predictions (stop-gradient on RGBD).
    
    This encourages the model to not over-rely on depth and learn robust
    representations that work with missing modalities (common in real data).
    """
    
    def __init__(
        self,
        dropout_prob: float = 0.3,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
    
    def forward(
        self,
        full_preds: Dict[str, torch.Tensor],
        dropped_preds: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute MSE consistency between full and dropped predictions.
        
        Args:
            full_preds: Dict with 'pred_logits' and 'pred_masks' from RGBD forward
            dropped_preds: Dict with 'pred_logits' and 'pred_masks' from RGB-only forward
            
        Returns:
            Scalar consistency loss (MSE on logits + masks)
        """
        # Logits consistency
        logits_loss = F.mse_loss(dropped_preds["pred_logits"], full_preds["pred_logits"].detach())
        
        # Mask consistency (use soft predictions, not binary)
        masks_loss = F.mse_loss(
            dropped_preds["pred_masks"].sigmoid(),
            full_preds["pred_masks"].sigmoid().detach()
        )
        
        return logits_loss + masks_loss


class UncertaintyWeighting(nn.Module):
    """Learnable loss weighting via log-variance (Kendall et al., NeurIPS 2018)."""
    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, *losses):
        total = self.log_vars.sum() * 0.0
        weighted = {}
        for i, loss in enumerate(losses):
            if loss is not None and isinstance(loss, torch.Tensor):
                precision = torch.exp(-self.log_vars[i])
                w = 0.5 * precision * loss + 0.5 * self.log_vars[i]
                total = total + w
                weighted[f"uw_task{i}"] = w.detach()
        return total, weighted
