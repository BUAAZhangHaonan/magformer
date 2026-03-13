# -*- coding: utf-8 -*-
"""
Simple Transformer Decoder

Minimal MaskFormer-style transformer decoder for single-class instance segmentation.
"""

from typing import Dict

import torch
import torch.nn as nn


def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """Build 2D sine-cosine positional embeddings."""
    if dim % 4 != 0:
        raise ValueError(
            "positional embedding dimension must be divisible by 4")

    y_embed = torch.linspace(0, 1, h, device=device)
    x_embed = torch.linspace(0, 1, w, device=device)
    yy, xx = torch.meshgrid(y_embed, x_embed, indexing="ij")

    dim_t = torch.arange(dim // 4, device=device, dtype=torch.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / (dim // 2))

    pos_x = xx[..., None] / dim_t
    pos_y = yy[..., None] / dim_t

    pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)

    pos = torch.cat((pos_y, pos_x), dim=-1)
    pos = pos.view(h * w, dim)
    return pos


class SimpleTransformerDecoder(nn.Module):
    """Minimal transformer decoder that predicts class logits and mask embeddings."""

    def __init__(
        self,
        num_queries: int,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        num_layers: int,
        num_classes: int,
        mask_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = nn.Linear(hidden_dim, mask_dim)

    def forward(
        self,
        memory: torch.Tensor,
        mask_features: torch.Tensor,
        multi_scale_features=None,
        multi_scale_pos=None,
    ) -> Dict[str, torch.Tensor]:
        del multi_scale_features, multi_scale_pos
        """
        Args:
            memory: (B, C, H, W) feature map for transformer memory
            mask_features: (B, mask_dim, Hm, Wm) feature map for mask prediction
        Returns:
            dict with pred_logits (B, Q, C) and pred_masks (B, Q, Hm, Wm)
        """
        b, c, h, w = memory.shape
        device = memory.device

        memory_flat = memory.flatten(2).permute(2, 0, 1)
        pos = _build_2d_sincos_pos_embed(h, w, c, device)
        pos = pos[:, None, :].expand(-1, b, -1)
        memory_flat = memory_flat + pos

        query_embed = self.query_embed.weight[:, None, :].repeat(1, b, 1)
        tgt = torch.zeros_like(query_embed)
        intermediate = []
        for layer in self.decoder.layers:
            tgt = layer(tgt + query_embed, memory_flat)
            intermediate.append(tgt.transpose(0, 1))

        hs = intermediate[-1]
        pred_logits = self.class_embed(hs)
        mask_embed = self.mask_embed(hs)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        aux_outputs = []
        for aux_hs in intermediate[:-1]:
            aux_logits = self.class_embed(aux_hs)
            aux_mask_embed = self.mask_embed(aux_hs)
            aux_masks = torch.einsum(
                "bqc,bchw->bqhw", aux_mask_embed, mask_features)
            aux_outputs.append(
                {"pred_logits": aux_logits, "pred_masks": aux_masks})

        return {"pred_logits": pred_logits, "pred_masks": pred_masks, "aux_outputs": aux_outputs}
