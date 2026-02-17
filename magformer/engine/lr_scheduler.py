# -*- coding: utf-8 -*-
"""Learning-rate scheduler utilities with Detectron2-like warmup behavior."""

from __future__ import annotations

from bisect import bisect_right
from typing import Iterable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_warmup_factor_at_iter(
    method: str,
    cur_iter: int,
    warmup_iters: int,
    warmup_factor: float,
) -> float:
    if warmup_iters <= 0 or cur_iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return float(warmup_factor)
    if method != "linear":
        raise ValueError(f"Unknown warmup method: {method}")

    alpha = float(cur_iter) / float(warmup_iters)
    return float(warmup_factor) * (1.0 - alpha) + alpha


def build_warmup_multistep_scheduler(
    optimizer: Optimizer,
    milestones: Iterable[int],
    gamma: float,
    warmup_iters: int,
    warmup_factor: float,
    warmup_method: str = "linear",
) -> LambdaLR:
    milestones = sorted(int(m) for m in milestones)

    def lr_lambda(iter_idx: int) -> float:
        warmup = _get_warmup_factor_at_iter(
            warmup_method,
            int(iter_idx),
            int(warmup_iters),
            float(warmup_factor),
        )
        decay = float(gamma) ** bisect_right(milestones, int(iter_idx))
        return warmup * decay

    return LambdaLR(optimizer, lr_lambda)


def build_warmup_poly_scheduler(
    optimizer: Optimizer,
    max_iter: int,
    warmup_iters: int,
    warmup_factor: float,
    warmup_method: str = "linear",
    power: float = 0.9,
) -> LambdaLR:
    max_iter = max(int(max_iter), 1)
    warmup_iters = max(int(warmup_iters), 0)

    def lr_lambda(iter_idx: int) -> float:
        warmup = _get_warmup_factor_at_iter(
            warmup_method,
            int(iter_idx),
            warmup_iters,
            float(warmup_factor),
        )

        if int(iter_idx) < warmup_iters:
            return warmup

        denom = max(max_iter - warmup_iters, 1)
        progress = float(int(iter_idx) - warmup_iters) / float(denom)
        progress = min(max(progress, 0.0), 1.0)
        poly = (1.0 - progress) ** float(power)
        return warmup * poly

    return LambdaLR(optimizer, lr_lambda)
