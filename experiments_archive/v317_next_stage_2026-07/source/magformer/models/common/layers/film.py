# -*- coding: utf-8 -*-
"""
Feature-wise Linear Modulation (FiLM) layer.

Provides conditional modulation of feature maps via gamma/beta
parameters predicted from a conditioning signal.
"""

import torch
from torch import nn


class FiLMModulation(nn.Module):
    """Feature-wise Linear Modulation with residual init.

    FiLM(x) = (1 + gamma) * x + beta
    At initialization gamma=0, beta=0, so output = x (identity).
    """

    def __init__(self, feature_dim, cond_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim * 2
        self.gamma_beta = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        # Identity init: gamma=0, beta=0 -> output = (1+0)*x + 0 = x
        nn.init.zeros_(self.gamma_beta[-1].weight)
        nn.init.zeros_(self.gamma_beta[-1].bias)

    def forward(self, features, cond):
        """
        Args:
            features: (B, C, H, W) feature map to modulate
            cond: (B, cond_dim) conditioning signal
        Returns:
            modulated: (B, C, H, W)
        """
        gamma_beta = self.gamma_beta(cond)  # (B, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # each (B, C)
        gamma = 1.0 + gamma  # residual: identity at init
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return gamma * features + beta
