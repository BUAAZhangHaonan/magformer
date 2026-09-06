"""
MP-Former Mask-Piloted Training Module for MagFormer.

Based on "MP-Former: Mask-Piloted Transformer for Image Segmentation" (CVPR 2023).
During training, injects noisy GT masks as pilot signals into the masked attention
mechanism of intermediate decoder layers. This gives each layer a strong starting
signal for cross-attention, instead of relying solely on predicted masks that are
inaccurate early in training.

The pilot signal replaces the predicted-mask-based attention mask with a noisy
version of the GT mask attention mask at each intermediate decoder layer.
During inference, the module is completely bypassed.

Expected improvement: +0.5-1.5 AP, zero inference cost, ~10% training time increase.

Usage in decoder forward():
    pilot = MaskPilotModule(num_layers=dec_layers)
    # In training loop, after computing predictions at each layer:
    if self.training and targets is not None:
        pilot_masks = pilot(gt_masks_list, layer_idx, num_queries, mask_size, device)
        # Use pilot_masks to override attention mask for next layer
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MaskPilotModule(nn.Module):
    """
    MP-Former Mask-Piloted Training Module.

    During training, generates noisy GT mask attention masks for intermediate
    decoder layers. These replace the predicted-mask-based attention masks,
    giving each decoder layer a strong signal about where objects are.

    The noise level decays linearly across layers: early layers get noisier
    pilots (harder task), later layers get cleaner pilots (easier refinement).
    This encourages each layer to learn genuine refinement behavior.

    During inference, this module returns None -- no pilot signals are used.

    Args:
        num_layers: Number of decoder layers (e.g., 9 for dec_layers=9).
        initial_noise_std: Noise standard deviation at first intermediate layer.
            Controls how much Gaussian noise is added to the GT mask logits.
            Default 1.0 (high noise, making early pilots quite imprecise).
        final_noise_std: Noise std at last intermediate layer.
            Default 0.1 (low noise, making later pilots close to GT).
        pilot_start_layer: Which decoder layer index to start injecting pilots.
            Default 1 (0-indexed). Layer 0 uses predicted masks as-is.
            Skipping layer 0 lets the first layer learn from scratch.
    """

    def __init__(
        self,
        num_layers: int,
        initial_noise_std: float = 1.0,
        final_noise_std: float = 0.1,
        pilot_start_layer: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.initial_noise_std = initial_noise_std
        self.final_noise_std = final_noise_std
        self.pilot_start_layer = pilot_start_layer

        if pilot_start_layer >= num_layers:
            logger.warning(
                f"[MaskPilot] pilot_start_layer ({pilot_start_layer}) >= num_layers "
                f"({num_layers}). No pilot injection will occur."
            )

    def get_noise_std(self, layer_idx: int) -> float:
        """
        Compute the noise standard deviation for a given decoder layer.

        Uses linear interpolation between initial_noise_std (at pilot_start_layer)
        and final_noise_std (at the last decoder layer).

        Args:
            layer_idx: Current decoder layer index (0-indexed).

        Returns:
            Noise standard deviation for this layer. Returns 0 if before
            pilot_start_layer.
        """
        if layer_idx < self.pilot_start_layer:
            return 0.0

        # Number of layers that actually get pilot signals
        pilot_range = self.num_layers - 1 - self.pilot_start_layer
        if pilot_range <= 0:
            return self.final_noise_std

        # Progress from 0.0 (at pilot_start_layer) to 1.0 (at last layer)
        progress = (layer_idx - self.pilot_start_layer) / pilot_range
        progress = min(max(progress, 0.0), 1.0)

        # Linear interpolation: high noise -> low noise
        noise_std = self.initial_noise_std + progress * (self.final_noise_std - self.initial_noise_std)
        return noise_std

    def generate_pilot_masks(
        self,
        gt_masks: torch.Tensor,
        layer_idx: int,
        target_size: tuple = None,
    ) -> torch.Tensor:
        """
        Generate noisy pilot masks from GT masks for a specific decoder layer.

        The process:
        1. Convert binary GT masks to logit space (0 -> -K, 1 -> +K)
        2. Add Gaussian noise with layer-dependent std
        3. Apply sigmoid to get soft masks
        4. Clamp to [0, 1]

        This approach preserves differentiability and avoids binarization
        artifacts that would lose gradient information.

        Args:
            gt_masks: (num_gt, H, W) binary GT masks (0.0 or 1.0).
            layer_idx: Current decoder layer index.
            target_size: (h, w) to resize masks to match decoder feature map
                resolution. If None, no resizing.

        Returns:
            pilot_masks: (num_gt, h, w) noisy soft masks in [0, 1].
        """
        noise_std = self.get_noise_std(layer_idx)

        # Resize GT masks to target resolution if needed
        if target_size is not None:
            masks = gt_masks.float().unsqueeze(0)  # (1, num_gt, H, W)
            masks = F.interpolate(
                masks,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            masks = masks.squeeze(0)  # (num_gt, h, w)
        else:
            masks = gt_masks.float()

        if noise_std <= 0.0:
            return masks

        # Convert binary masks to logit space for noise injection
        # Binary 0/1 -> logits: 0 maps to -logit_k, 1 maps to +logit_k
        # Using K=5 gives sigmoid(K) ~ 0.993, close to binary
        logit_k = 5.0
        mask_logits = (masks * 2.0 - 1.0) * logit_k  # 0 -> -5, 1 -> +5

        # Add Gaussian noise
        noise = torch.randn_like(mask_logits) * noise_std
        noisy_logits = mask_logits + noise

        # Back to [0, 1] via sigmoid
        pilot_masks = noisy_logits.sigmoid()

        return pilot_masks

    @torch.no_grad()
    def forward(
        self,
        gt_masks_list: list,
        layer_idx: int,
        num_queries: int,
        target_size: tuple,
        num_heads: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate pilot attention masks for a decoder layer during training.

        For each image in the batch:
        1. Generate noisy pilot masks from GT masks
        2. Convert to boolean attention masks (True = masked/blocked)
        3. Create a (num_queries, h*w) mask where:
           - All positions start as unmasked (False)
           - The pilot masks override with GT-based masks

        The output is formatted as [B * num_heads, total_queries, h*w] to
        match the attention mask format used in Mask2Former's cross-attention.

        Important: This only produces pilot masks for the GT object positions.
        The caller should merge these with the predicted attention masks for
        the non-GT queries. A common pattern is:
        - For the first `num_gt` query slots: use pilot mask
        - For remaining query slots: use predicted mask

        However, in practice the pilot replaces the entire attention mask
        because the Hungarian matching has already assigned queries to GT.
        We generate per-image masks padded to max GT count.

        Args:
            gt_masks_list: List of tensors, one per image in the batch.
                Each tensor has shape (num_gt_i, H, W) with binary GT masks.
            layer_idx: Decoder layer index.
            num_queries: Total number of object queries.
            target_size: (h, w) spatial resolution of the feature map at
                this decoder layer.
            num_heads: Number of attention heads (for reshaping).
            device: Torch device.

        Returns:
            pilot_attn_mask: (B * num_heads, num_queries, h*w) boolean tensor.
                True positions are masked (blocked from attention).
                Returns None if layer_idx < pilot_start_layer or if
                no GT masks are available.
        """
        if layer_idx < self.pilot_start_layer:
            return None

        batch_size = len(gt_masks_list)

        # Check if we have any GT masks at all
        total_gt = sum(m.shape[0] for m in gt_masks_list)
        if total_gt == 0:
            return None

        h, w = target_size
        hw = h * w

        # Build pilot attention masks per image
        pilot_masks_batch = []

        for gt_masks in gt_masks_list:
            num_gt = gt_masks.shape[0]

            if num_gt == 0:
                # No GT objects: all positions unmasked (False)
                pilot_masks_batch.append(
                    torch.zeros(num_queries, hw, dtype=torch.bool, device=device)
                )
                continue

            # Generate noisy pilot masks: (num_gt, h, w)
            pilot = self.generate_pilot_masks(gt_masks, layer_idx, target_size)

            # Convert to attention mask format:
            # sigmoid > 0.5 means object present -> False (attend)
            # sigmoid <= 0.5 means no object -> True (masked/blocked)
            # Shape: (num_gt, h*w)
            pilot_attn = (pilot <= 0.5).reshape(num_gt, hw)

            # Pad to num_queries rows with False (attend to all)
            full_mask = torch.zeros(num_queries, hw, dtype=torch.bool, device=device)
            rows_to_fill = min(num_gt, num_queries)
            full_mask[:rows_to_fill] = pilot_attn[:rows_to_fill]

            pilot_masks_batch.append(full_mask)

        # Stack: (B, num_queries, h*w)
        pilot_masks_batch = torch.stack(pilot_masks_batch, dim=0)

        # Expand for multi-head attention: (B, num_heads, num_queries, h*w)
        pilot_masks_batch = pilot_masks_batch.unsqueeze(1).expand(
            -1, num_heads, -1, -1
        )

        # Flatten to (B * num_heads, num_queries, h*w)
        pilot_masks_batch = pilot_masks_batch.reshape(
            batch_size * num_heads, num_queries, hw
        )

        return pilot_masks_batch

    def generate_pilot_masks_for_all_layers(
        self,
        gt_masks_list: list,
        num_queries: int,
        num_heads: int,
        device: torch.device,
        mask_feature_size: tuple,
    ) -> dict:
        """
        Pre-generate pilot masks for all layers at once.

        This is more efficient than calling forward() per layer because
        the GT mask resizing only happens once. The noise injection is
        still per-layer.

        Args:
            gt_masks_list: List of (num_gt_i, H, W) binary GT masks per image.
            num_queries: Number of object queries.
            num_heads: Number of attention heads.
            device: Torch device.
            mask_feature_size: (h, w) resolution of the mask features.

        Returns:
            Dictionary mapping layer_idx -> (B * num_heads, num_queries, h*w)
            attention mask tensor. Only layers >= pilot_start_layer are included.
        """
        total_gt = sum(m.shape[0] for m in gt_masks_list)
        if total_gt == 0:
            return {}

        h, w = mask_feature_size
        hw = h * w
        batch_size = len(gt_masks_list)

        # Pre-resize GT masks once
        resized_gt_masks = []
        for gt_masks in gt_masks_list:
            if gt_masks.shape[0] == 0:
                resized_gt_masks.append(None)
                continue
            resized = F.interpolate(
                gt_masks.float().unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            resized_gt_masks.append(resized)

        pilot_masks_by_layer = {}

        for layer_idx in range(self.pilot_start_layer, self.num_layers):
            noise_std = self.get_noise_std(layer_idx)
            pilot_masks_batch = []

            for img_idx, resized in enumerate(resized_gt_masks):
                if resized is None:
                    pilot_masks_batch.append(
                        torch.zeros(num_queries, hw, dtype=torch.bool, device=device)
                    )
                    continue

                num_gt = resized.shape[0]

                # Apply noise injection in logit space
                logit_k = 5.0
                mask_logits = (resized * 2.0 - 1.0) * logit_k
                noise = torch.randn_like(mask_logits) * noise_std
                noisy_logits = mask_logits + noise
                pilot = noisy_logits.sigmoid()

                # Convert to attention mask
                pilot_attn = (pilot <= 0.5).reshape(num_gt, hw)

                full_mask = torch.zeros(num_queries, hw, dtype=torch.bool, device=device)
                rows_to_fill = min(num_gt, num_queries)
                full_mask[:rows_to_fill] = pilot_attn[:rows_to_fill]

                pilot_masks_batch.append(full_mask)

            # Stack and reshape: (B * num_heads, num_queries, h*w)
            pilot_tensor = torch.stack(pilot_masks_batch, dim=0)
            pilot_tensor = pilot_tensor.unsqueeze(1).expand(
                -1, num_heads, -1, -1
            )
            pilot_tensor = pilot_tensor.reshape(
                batch_size * num_heads, num_queries, hw
            )

            pilot_masks_by_layer[layer_idx] = pilot_tensor

        return pilot_masks_by_layer

    def state_dict(self, *args, **kwargs):
        """No learnable parameters, but include config for checkpointing."""
        return {
            "num_layers": self.num_layers,
            "initial_noise_std": self.initial_noise_std,
            "final_noise_std": self.final_noise_std,
            "pilot_start_layer": self.pilot_start_layer,
        }

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Restore config from checkpoint."""
        self.num_layers = state_dict.get("num_layers", self.num_layers)
        self.initial_noise_std = state_dict.get("initial_noise_std", self.initial_noise_std)
        self.final_noise_std = state_dict.get("final_noise_std", self.final_noise_std)
        self.pilot_start_layer = state_dict.get("pilot_start_layer", self.pilot_start_layer)
        logger.info(
            f"[MaskPilot] Loaded config: layers={self.num_layers}, "
            f"noise=[{self.initial_noise_std} -> {self.final_noise_std}], "
            f"start_layer={self.pilot_start_layer}"
        )
