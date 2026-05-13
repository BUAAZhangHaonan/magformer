# -*- coding: utf-8 -*-
"""Pseudo-Label Quality Scorer for VC-SUDA.

Scores teacher predictions using a 5-factor quality metric:
1. Class confidence (softmax max probability)
2. Mask confidence (mean sigmoid probability)  
3. Mask stability (IoU between soft and hard masks)
4. Depth boundary sharpness (Sobel gradient at mask boundary)
5. Overlap penalty (1 - max pairwise IoU)

Final quality: q_i = s_cls * mask_mean_prob * stability_IoU * B_depth * (1 - overlap)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional


class PseudoLabelScorer:
    """
    Scores teacher outputs for pseudo-label quality.
    
    Each prediction gets a quality score in [0, 1]. Higher = more reliable.
    Used by the curriculum scheduler to filter noisy pseudo-labels.
    """
    
    def __init__(
        self,
        max_instances: int = 100,
        sobel_kernel_size: int = 3,
        min_mask_area_pixels: int = 1,
        min_mask_area_ratio: float = 1e-4,
        mask_topk_ratio: float = 0.01,
    ):
        """
        Args:
            max_instances: Max instances to keep per image
            sobel_kernel_size: Kernel size for Sobel gradient computation
            min_mask_area_pixels: Absolute lower bound for a valid mask
            min_mask_area_ratio: Image-relative lower bound for a valid mask
            mask_topk_ratio: Fraction of pixels used when no hard foreground exists
        """
        self.max_instances = max_instances
        self.sobel_kernel_size = sobel_kernel_size
        self.min_mask_area_pixels = min_mask_area_pixels
        self.min_mask_area_ratio = min_mask_area_ratio
        self.mask_topk_ratio = mask_topk_ratio
        self._sobel_x = None
        self._sobel_y = None
    
    def _get_sobel_kernels(self, device: torch.device) -> tuple:
        """Get Sobel kernels, creating them lazily on first use."""
        if self._sobel_x is None or self._sobel_x.device != device:
            # Sobel kernels for gradient computation
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32, device=device
            ).reshape(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=torch.float32, device=device
            ).reshape(1, 1, 3, 3)
            self._sobel_x = sobel_x
            self._sobel_y = sobel_y
        return self._sobel_x, self._sobel_y
    
    def _compute_mask_boundary(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute mask boundary via dilation - erosion.
        
        Args:
            masks: (N, H, W) binary masks
            
        Returns:
            (N, H, W) boundary mask
        """
        # Simple boundary: gradient of binary mask
        masks_float = masks.float().unsqueeze(1)  # (N, 1, H, W)
        padded = F.pad(masks_float, [1, 1, 1, 1], mode='constant', value=0)
        
        # Max pooling (dilation) - manual 3x3
        dilated = F.max_pool2d(padded, 3, stride=1)
        # Min pooling (erosion) via -max_pool(-x)
        eroded = -F.max_pool2d(-padded, 3, stride=1)
        
        # Boundary = dilated - eroded
        boundary = (dilated - eroded).squeeze(1)  # (N, H, W)
        return (boundary > 0).float()
    
    def _compute_depth_boundary_score(
        self,
        depth: torch.Tensor,
        mask_boundary: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute depth gradient magnitude at mask boundaries.
        
        Args:
            depth: (1, H, W) or (H, W) depth map
            mask_boundary: (N, H, W) boundary masks
            
        Returns:
            (N,) boundary sharpness score in [0, 1]
        """
        if depth.ndim == 3:
            depth = depth.unsqueeze(0)  # (1, 1, H, W)
        elif depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        sobel_x, sobel_y = self._get_sobel_kernels(depth.device)
        grad_x = F.conv2d(depth, sobel_x, padding=1)
        grad_y = F.conv2d(depth, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # (1, 1, H, W)
        grad_mag = grad_mag.squeeze(0).squeeze(0)  # (H, W)
        
        # Average gradient at boundary pixels
        N = mask_boundary.shape[0]
        scores = []
        for i in range(N):
            boundary_pixels = mask_boundary[i]  # (H, W)
            if boundary_pixels.sum() > 0:
                avg_grad = (grad_mag * boundary_pixels).sum() / (boundary_pixels.sum() + 1e-6)
                # Normalize to [0, 1] using sigmoid-like transform
                score = torch.sigmoid(avg_grad / 10.0 - 1.0)
            else:
                score = torch.tensor(0.5, device=depth.device)
            scores.append(score)
        
        return torch.stack(scores)  # (N,)
    
    def _compute_overlap_penalty(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise overlap penalty.
        
        Args:
            masks: (N, H, W) binary masks
            
        Returns:
            (N,) overlap penalty: 1 - max pairwise IoU with other masks
        """
        N = masks.shape[0]
        if N <= 1:
            return torch.ones(N, device=masks.device)
        
        # Flatten masks for IoU computation
        masks_flat = masks.view(N, -1).float()  # (N, H*W)
        
        # Pairwise intersection
        intersection = masks_flat @ masks_flat.T  # (N, N)
        areas = masks_flat.sum(dim=1, keepdim=True)  # (N, 1)
        union = areas + areas.T - intersection  # (N, N)
        iou = intersection / (union + 1e-6)  # (N, N)
        
        # Zero out self-IoU
        iou.fill_diagonal_(0)
        
        # Max IoU with any other mask
        max_iou, _ = iou.max(dim=1)  # (N,)
        
        return 1.0 - max_iou  # (N,) penalty: 1 = no overlap, 0 = full overlap
    
    @torch.no_grad()
    def score(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        depth_maps: torch.Tensor,
        image_shape: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        Score all teacher predictions for pseudo-label quality.
        
        Args:
            teacher_outputs: Dict with 'pred_logits' (B, Nq, C) and 'pred_masks' (B, Nq, H, W)
            depth_maps: (B, 1, H_orig, W_orig) depth maps
            image_shape: Original image size (H, W) for mask resizing
            
        Returns:
            List of length B, each element is a dict with:
                'scores': (N_filtered,) quality scores
                'labels': (N_filtered,) predicted class labels
                'masks': (N_filtered, H, W) binary masks
                'logits': (N_filtered, C) class logits
        """
        pred_logits = teacher_outputs["pred_logits"]  # (B, Nq, C)
        pred_masks = teacher_outputs["pred_masks"]  # (B, Nq, H, W)
        
        B, Nq, C = pred_logits.shape
        H_mask, W_mask = pred_masks.shape[-2:]
        
        results = []
        for b in range(B):
            logits_b = pred_logits[b]  # (Nq, C)
            masks_b = pred_masks[b]  # (Nq, H, W)
            depth_b = depth_maps[b]  # (1, H_orig, W_orig)
            
            # Resize depth to mask size if needed
            if depth_b.shape[-2:] != (H_mask, W_mask):
                depth_resized = F.interpolate(
                    depth_b.unsqueeze(0), size=(H_mask, W_mask),
                    mode='bilinear', align_corners=False
                ).squeeze(0)  # (1, H_mask, W_mask)
            else:
                depth_resized = depth_b
            
            # 1. Class confidence
            class_probs = F.softmax(logits_b, dim=-1)
            # For single-class (num_classes=1), take foreground prob
            # logits shape is (Nq, C) where C includes background
            if C > 1:
                fg_probs = class_probs[:, :-1]  # exclude background
                s_cls = fg_probs.max(dim=-1).values  # (Nq,)
                labels = fg_probs.argmax(dim=-1)  # (Nq,)
            else:
                # Binary: single foreground class, C=1
                s_cls = class_probs[:, 0]  # (Nq,)
                labels = torch.zeros(Nq, dtype=torch.long, device=logits_b.device)
            
            # 2. Mask confidence
            mask_probs = masks_b.sigmoid()  # (Nq, H, W)
            hard_masks = (mask_probs > 0.5).float()  # (Nq, H, W)
            hard_areas = hard_masks.flatten(1).sum(dim=1)
            fg_confidence = (mask_probs * hard_masks).flatten(1).sum(dim=1) / hard_areas.clamp_min(1.0)
            topk_count = max(1, int(mask_probs.shape[-2] * mask_probs.shape[-1] * self.mask_topk_ratio))
            topk_confidence = mask_probs.flatten(1).topk(topk_count, dim=1).values.mean(dim=1)
            mask_confidence = torch.where(hard_areas > 0, fg_confidence, topk_confidence)
            
            # 3. Mask stability (IoU between soft and hard masks)
            intersection = (mask_probs * hard_masks).flatten(1).sum(dim=1)
            union = mask_probs.flatten(1).sum(dim=1) + hard_masks.flatten(1).sum(dim=1) - intersection
            stability_iou = intersection / (union + 1e-6)  # (Nq,)
            
            # 4. Depth boundary score
            boundary_score = self._compute_depth_boundary_score(
                depth_resized, hard_masks
            )  # (Nq,)
            
            # 5. Overlap penalty
            overlap_penalty = self._compute_overlap_penalty(hard_masks)  # (Nq,)
            
            # Combined quality score
            quality = s_cls * mask_confidence * stability_iou * boundary_score * overlap_penalty  # (Nq,)
            
            # Filter: keep top max_instances by quality
            # Only keep instances with meaningful masks
            min_effective_area = max(
                float(self.min_mask_area_pixels),
                float(H_mask * W_mask) * float(self.min_mask_area_ratio),
            )
            valid_mask = hard_areas >= min_effective_area
            
            if valid_mask.sum() == 0:
                results.append({
                    'scores': torch.zeros(0, device=logits_b.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=logits_b.device),
                    'masks': torch.zeros(0, H_mask, W_mask, device=logits_b.device),
                    'logits': torch.zeros(0, C, device=logits_b.device),
                })
                continue
            
            # Get valid indices sorted by quality
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            valid_quality = quality[valid_indices]
            
            # Sort by quality descending
            sorted_indices = valid_quality.argsort(descending=True)
            keep_count = min(self.max_instances, len(sorted_indices))
            top_indices = valid_indices[sorted_indices[:keep_count]]
            
            results.append({
                'scores': quality[top_indices],
                'labels': labels[top_indices],
                'masks': mask_probs[top_indices],  # soft masks for student training
                'logits': logits_b[top_indices],
            })
        
        return results
    
    def filter_by_threshold(
        self,
        scored_results: List[Dict[str, Any]],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Filter scored results by quality threshold.
        
        Args:
            scored_results: Output from score()
            threshold: Minimum quality score
            
        Returns:
            Filtered results with same structure
        """
        filtered = []
        for result in scored_results:
            scores = result['scores']
            if len(scores) == 0:
                filtered.append(result)
                continue
            
            keep = scores >= threshold
            filtered.append({
                'scores': scores[keep],
                'labels': result['labels'][keep],
                'masks': result['masks'][keep],
                'logits': result['logits'][keep],
            })
        
        return filtered
