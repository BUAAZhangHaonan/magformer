"""Query-preserving RGB-D point refinement for instance masks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def point_sample(input_tensor: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    """Sample ``input_tensor`` at normalized ``[0, 1]`` point coordinates."""
    grid = point_coords * 2.0 - 1.0
    sampled = F.grid_sample(
        input_tensor,
        grid.unsqueeze(2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled.squeeze(-1)


def calculate_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """Binary-mask uncertainty used by PointRend."""
    return -torch.abs(logits)


def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> torch.Tensor:
    """PointRend training coordinates: uncertain points plus random coverage."""
    num_masks = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(
        num_masks, num_sampled, 2, device=coarse_logits.device
    )
    point_logits = point_sample(coarse_logits, point_coords).squeeze(1)
    uncertainties = calculate_uncertainty(point_logits)
    num_uncertain = int(importance_sample_ratio * num_points)
    num_random = num_points - num_uncertain

    indices = uncertainties.topk(k=num_uncertain, dim=1).indices
    uncertain_coords = torch.gather(
        point_coords,
        1,
        indices.unsqueeze(-1).expand(-1, -1, 2),
    )
    if num_random == 0:
        return uncertain_coords
    random_coords = torch.rand(
        num_masks, num_random, 2, device=coarse_logits.device
    )
    return torch.cat((uncertain_coords, random_coords), dim=1)


class RGBDPointRefiner(nn.Module):
    """Refine each query mask with shared, high-resolution RGB-D evidence."""

    detail_dim = 32
    hidden_dim = 64

    def __init__(
        self,
        train_num_points: int,
        subdivision_steps: int,
        subdivision_num_points: int,
    ) -> None:
        super().__init__()
        self.train_num_points = int(train_num_points)
        self.subdivision_steps = int(subdivision_steps)
        self.subdivision_num_points = int(subdivision_num_points)
        if self.train_num_points <= 0:
            raise ValueError("train_num_points must be positive")
        if self.subdivision_steps != 2:
            raise ValueError(
                "RGB-D point refinement requires exactly two subdivision steps"
            )
        if self.subdivision_num_points <= 0:
            raise ValueError("subdivision_num_points must be positive")

        self.detail_stem = nn.Sequential(
            nn.Conv2d(5, self.detail_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, self.detail_dim),
            nn.GELU(),
        )
        self.point_mlp = nn.Sequential(
            nn.Conv1d(self.detail_dim + 1, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.point_mlp[-1].weight)
        nn.init.zeros_(self.point_mlp[-1].bias)

    def build_detail_features(
        self,
        normalized_rgb: torch.Tensor,
        depth: torch.Tensor,
        depth_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if depth_valid_mask is None:
            raise ValueError("RGB-D point refinement requires depth_valid_mask")
        if normalized_rgb.ndim != 4 or normalized_rgb.shape[1] != 3:
            raise ValueError(
                "normalized_rgb must have shape (B,3,H,W), "
                f"got {tuple(normalized_rgb.shape)}"
            )
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(
                f"depth must have shape (B,1,H,W), got {tuple(depth.shape)}"
            )
        if depth_valid_mask.shape != depth.shape:
            raise ValueError(
                "depth_valid_mask must exactly match depth shape: "
                f"mask={tuple(depth_valid_mask.shape)}, depth={tuple(depth.shape)}"
            )
        if depth_valid_mask.dtype is not torch.bool:
            raise TypeError(
                f"depth_valid_mask must be bool, got {depth_valid_mask.dtype}"
            )
        if normalized_rgb.shape[0] != depth.shape[0] or normalized_rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError(
                "normalized RGB and depth geometry must match: "
                f"rgb={tuple(normalized_rgb.shape)}, depth={tuple(depth.shape)}"
            )
        if normalized_rgb.device != depth.device or depth_valid_mask.device != depth.device:
            raise ValueError("normalized RGB, depth, and valid mask must share a device")
        valid = depth_valid_mask.to(dtype=depth.dtype)
        valid_depth = torch.where(depth_valid_mask, depth, torch.zeros_like(depth))
        detail_input = torch.cat((normalized_rgb, valid_depth, valid), dim=1)
        return self.detail_stem(detail_input)

    @staticmethod
    def _sample_detail_by_image(
        detail_features: torch.Tensor,
        point_coords: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_masks, num_points = point_coords.shape[:2]
        channels = detail_features.shape[1]
        sampled = detail_features.new_empty((num_masks, channels, num_points))
        for image_index in range(detail_features.shape[0]):
            mask_indices = torch.nonzero(
                batch_indices == image_index, as_tuple=False
            ).flatten()
            if mask_indices.numel() == 0:
                continue
            image_coords = point_coords.index_select(0, mask_indices).reshape(1, -1, 2)
            image_samples = point_sample(
                detail_features[image_index : image_index + 1], image_coords
            )
            image_samples = image_samples.reshape(
                channels, mask_indices.numel(), num_points
            ).permute(1, 0, 2)
            sampled.index_copy_(0, mask_indices, image_samples)
        return sampled

    def forward(
        self,
        coarse_point_logits: torch.Tensor,
        detail_features: torch.Tensor,
        point_coords: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        detail_points = self._sample_detail_by_image(
            detail_features, point_coords, batch_indices
        )
        delta = self.point_mlp(torch.cat((coarse_point_logits, detail_points), dim=1))
        return coarse_point_logits + delta

    @staticmethod
    def _uncertain_indices(logits: torch.Tensor, num_points: int) -> torch.Tensor:
        uncertainty = calculate_uncertainty(logits).flatten(1)
        return uncertainty.topk(
            k=min(int(num_points), uncertainty.shape[1]), dim=1
        ).indices

    @staticmethod
    def _indices_to_coords(
        indices: torch.Tensor,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        x = (indices.remainder(width).to(dtype) + 0.5) / float(width)
        y = (indices.div(width, rounding_mode="floor").to(dtype) + 0.5) / float(height)
        return torch.stack((x, y), dim=-1)

    def subdivide(
        self,
        coarse_masks: torch.Tensor,
        detail_features: torch.Tensor,
    ) -> torch.Tensor:
        """Run PointRend subdivisions after inference top-k selection."""
        batch_size, num_queries = coarse_masks.shape[:2]
        masks = coarse_masks
        batch_indices = torch.arange(
            batch_size, device=masks.device
        ).repeat_interleave(num_queries)

        for _ in range(self.subdivision_steps):
            previous_height, previous_width = masks.shape[-2:]
            masks = F.interpolate(
                masks.flatten(0, 1).unsqueeze(1),
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            ).reshape(
                batch_size,
                num_queries,
                previous_height * 2,
                previous_width * 2,
            )
            flat_masks = masks.flatten(0, 1).unsqueeze(1)
            point_indices = self._uncertain_indices(
                flat_masks, self.subdivision_num_points
            )
            point_coords = self._indices_to_coords(
                point_indices,
                flat_masks.shape[-2],
                flat_masks.shape[-1],
                flat_masks.dtype,
            )
            coarse_points = point_sample(flat_masks, point_coords)
            refined_points = self(
                coarse_points,
                detail_features,
                point_coords,
                batch_indices,
            )
            updated = flat_masks.flatten(2).clone()
            updated.scatter_(2, point_indices.unsqueeze(1), refined_points)
            masks = updated.reshape(
                batch_size,
                num_queries,
                flat_masks.shape[-2],
                flat_masks.shape[-1],
            )
        return masks
