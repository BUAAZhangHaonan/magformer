# -*- coding: utf-8 -*-
"""Transformer utility functions."""

import copy
import torch.nn.functional as F
from torch import nn


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(
        f"activation should be relu/gelu/glu, not {activation}.")
