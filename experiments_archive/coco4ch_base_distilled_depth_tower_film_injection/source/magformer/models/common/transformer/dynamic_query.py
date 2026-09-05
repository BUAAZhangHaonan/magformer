# -*- coding: utf-8 -*-
"""
Dynamic Content Query Module (DCQM).

PaQ-DETR inspired: replaces static query_feat embeddings with
image-conditioned dynamic queries formed from shared learnable patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicContentQueryModule(nn.Module):
    """
    Generates per-image dynamic content queries from a learnable pattern bank,
    conditioned on encoder features at each scale level.

    Architecture:
        1. Pattern bank: K shared content prototypes (learnable basis)
        2. Per-scale context extractors: pool encoder features -> context vectors
        3. Cross-scale fusion: merge contexts into a single image descriptor
        4. Pattern weight generation: base affinity + context modulation -> softmax
        5. Dynamic queries: weighted sum of patterns
        6. Output projection: refine dynamic queries

    ~900K parameters. Zero inference overhead when disabled.
    """

    def __init__(
        self,
        num_queries: int = 200,
        hidden_dim: int = 256,
        num_feature_levels: int = 3,
        num_patterns: int = 224,
        context_dim: int = 256,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        self.num_levels = num_feature_levels

        # Shared pattern bank: K content prototypes
        self.pattern_bank = nn.Embedding(num_patterns, hidden_dim)
        nn.init.normal_(self.pattern_bank.weight, std=0.02)

        # Per-scale context extractors: pool + 2-layer MLP
        self.context_nets = nn.ModuleList()
        for _ in range(num_feature_levels):
            self.context_nets.append(nn.Sequential(
                nn.Linear(hidden_dim, context_dim),
                nn.GELU(),
                nn.Linear(context_dim, context_dim),
            ))

        # Cross-scale fusion: merge all scale contexts
        self.fusion = nn.Sequential(
            nn.Linear(num_feature_levels * context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable base query-pattern affinity
        self.base_affinity = nn.Parameter(
            torch.randn(num_queries, num_patterns) * 0.02
        )

        # Context -> pattern modulation
        self.context_to_pattern = nn.Linear(hidden_dim, num_patterns)

        # Output projection with bottleneck
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Gating parameter: controls dynamic vs static content
        # Initialized to 0 so sigmoid(0)=0.5, balanced start
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, projected_features, static_query_feat, padding_masks):
        """
        Args:
            projected_features: list of [B, C, H_i, W_i] per scale level
                               (after input_proj, before level_embed)
            static_query_feat: (Q, C) original static query embeddings
            padding_masks: list of boolean (B, H_i, W_i), True for padding

        Returns:
            dynamic_queries: (Q, B, C) to use as decoder content queries
        """
        B = projected_features[0].shape[0]
        if len(padding_masks) != len(projected_features):
            raise ValueError(
                f"padding mask count {len(padding_masks)} must match feature count "
                f"{len(projected_features)}"
            )

        # 1. Extract per-scale context via valid-pixel average pooling
        contexts = []
        for i, (feat, mask) in enumerate(zip(projected_features, padding_masks)):
            if mask.dtype != torch.bool:
                raise TypeError(f"padding mask at level {i} must be bool")
            if mask.device != feat.device or mask.shape != (
                feat.shape[0], feat.shape[2], feat.shape[3]
            ):
                raise ValueError(
                    f"padding mask at level {i} is not aligned with its feature"
                )
            valid = (~mask).unsqueeze(1).to(feat.dtype)
            valid_count = valid.sum(dim=[2, 3])
            if (valid_count == 0).any():
                raise ValueError(f"all-padding feature level {i} is invalid for DCQM")
            pooled = (feat * valid).sum(dim=[2, 3]) / valid_count
            ctx = self.context_nets[i](pooled)  # (B, context_dim)
            contexts.append(ctx)

        # 2. Fuse cross-scale context
        fused = self.fusion(torch.cat(contexts, dim=-1))  # (B, hidden_dim)

        # 3. Generate pattern weights: base affinity + context modulation
        pattern_mod = self.context_to_pattern(fused)  # (B, num_patterns)
        logits = self.base_affinity.unsqueeze(0) + pattern_mod.unsqueeze(1)  # (B, Q, K)
        weights = F.softmax(logits, dim=-1)  # (B, Q, K)

        # 4. Dynamic queries: weighted sum over pattern bank
        dynamic_q = torch.einsum(
            "bqk,kd->bqd", weights, self.pattern_bank.weight
        )  # (B, Q, D)

        # 5. Output refinement
        dynamic_q = self.output_proj(dynamic_q)  # (B, Q, D)

        # 6. Gate: blend dynamic and static content
        g = torch.sigmoid(self.gate)
        static = static_query_feat.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)
        output = g * dynamic_q + (1 - g) * static  # (B, Q, D)

        return output.permute(1, 0, 2)  # (Q, B, C)
