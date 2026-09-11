# -*- coding: utf-8 -*-
"""2D sinusoidal position encoding."""

import math
from typing import Optional

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, _, h, w = x.shape
        if mask is None:
            mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)
        else:
            if mask.dtype != torch.bool:
                raise TypeError(f"position mask must be bool, got {mask.dtype}")
            if mask.device != x.device:
                raise ValueError(
                    f"position mask device {mask.device} must match input device {x.device}"
                )
            if mask.shape != (b, h, w):
                raise ValueError(
                    f"position mask shape {tuple(mask.shape)} must be {(b, h, w)}"
                )
            # Validation-only host sync (`.any()` -> bool): illegal during CUDA
            # graph capture. Shapes/values are data-independent across replays
            # for fixed-size eval inputs, so skip the guard while capturing.
            if not torch.cuda.is_current_stream_capturing():
                all_padding = mask.flatten(1).all(dim=1)
                if all_padding.any():
                    indices = all_padding.nonzero(as_tuple=False).flatten().tolist()
                    raise ValueError(
                        f"position mask contains all-padding samples: {indices}"
                    )

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_denominator = y_embed.amax(dim=1, keepdim=True)
            x_denominator = x_embed.amax(dim=2, keepdim=True)
            y_embed = y_embed / (y_denominator + eps) * self.scale
            x_embed = x_embed / (x_denominator + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.masked_fill(mask.unsqueeze(1), 0)
