# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


def normalize_and_augment_depth(depth: np.ndarray, cfg) -> np.ndarray:
    """
    输入: HxW 或 HxWx1
    输出: HxWx1, float32
    步骤: to float32 -> scale/shift -> clip -> normalize(按要求)
    """
    if depth.ndim == 2:
        depth = depth[..., None]
    elif not (depth.ndim == 3 and depth.shape[2] == 1):
        raise ValueError(f"Unexpected depth shape: {depth.shape}")

    depth = depth.astype(np.float32)
    depth = depth * float(cfg.INPUT.DEPTH_SCALE) + float(cfg.INPUT.DEPTH_SHIFT)

    dmin = float(cfg.INPUT.DEPTH_CLIP_MIN)
    dmax = float(cfg.INPUT.DEPTH_CLIP_MAX)
    per_sample_norm = bool(getattr(cfg.INPUT, "DEPTH_PER_SAMPLE_NORM", False))

    if not (dmax > dmin):
        return depth.astype(np.float32)

    depth = np.clip(depth, dmin, dmax)

    if per_sample_norm:
        sample_min = float(depth.min())
        sample_max = float(depth.max())
        if sample_max - sample_min > 1e-6:
            norm = (depth - sample_min) / (sample_max - sample_min)
        else:
            norm = np.zeros_like(depth, dtype=np.float32)
        norm = np.clip(norm, 0.0, 1.0)
    elif dmin == 0.0 and dmax == 1.0:
        return depth.astype(np.float32)
    else:
        norm = (depth - dmin) / (dmax - dmin + 1e-6)
        norm = np.clip(norm, 0.0, 1.0)

    depth_norm = cfg.INPUT.DEPTH_NORM
    if isinstance(depth_norm, (list, tuple)) and len(depth_norm) == 2:
        a, b = float(depth_norm[0]), float(depth_norm[1])
        norm = norm * (b - a) + a
        norm = np.clip(norm, min(a, b), max(a, b))

    return norm.astype(np.float32)

