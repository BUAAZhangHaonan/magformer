# -*- coding: utf-8 -*-
"""
Multi-Scale Deformable Attention CUDA operators.

This module provides:
1. CUDA-optimized MSDeformAttn (if compiled)
2. Pure PyTorch fallback implementation

To compile the CUDA extension:
    cd magformer/models/ops
    python setup.py build install
    # or
    sh make.sh
"""

from .modules import MSDeformAttn
from .functions import ms_deform_attn_core_pytorch

__all__ = ['MSDeformAttn', 'ms_deform_attn_core_pytorch']
