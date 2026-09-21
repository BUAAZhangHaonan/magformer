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


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    max_iter: int,
    warmup_iters: int,
    warmup_factor: float,
    warmup_method: str = "linear",
    hires_mult_anneal: bool = False,
    hires_anneal_start_frac: float = 0.1,
    hires_anneal_end_frac: float = 0.5,
) -> LambdaLR:
    import math
    max_iter = max(int(max_iter), 1)
    warmup_iters = max(int(warmup_iters), 0)

    def lr_lambda(iter_idx: int) -> float:
        warmup = _get_warmup_factor_at_iter(
            warmup_method,
            int(iter_idx),
            int(warmup_iters),
            float(warmup_factor),
        )

        if int(iter_idx) < warmup_iters:
            return warmup

        denom = max(max_iter - warmup_iters, 1)
        progress = float(int(iter_idx) - warmup_iters) / float(denom)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return warmup * cosine

    # HDA+ (arena P3-c winner): param groups tagged with a `hires_mult`
    # marker (their base lr has the multiplier baked in) get a
    # differentiated schedule whose effective multiplier anneals
    # hires_mult -> 1 over [start_frac, end_frac] of max_iter. Groups
    # without the marker keep the plain lambda object above, bit-identical
    # to the pre-HDA schedule. Previously this anneal lived only in the
    # never-imported engine/cosine_scheduler.py, so hires_mult_anneal
    # configs silently ran a constant multiplier (final-review finding).
    base_lambda = lr_lambda  # the closure below must call the PLAIN lambda;
    # binding lr_lambda itself would capture the later list reassignment.
    lambdas = []
    for group in optimizer.param_groups:
        m = group.get("hires_mult", 1.0)
        if hires_mult_anneal and m and float(m) > 1.0:
            m = float(m)
            s_frac = float(hires_anneal_start_frac)
            e_frac = float(hires_anneal_end_frac)
            t0, t1 = s_frac * max_iter, e_frac * max_iter

            def _hires_lambda(iter_idx: int, m=m, t0=t0, t1=t1, base=base_lambda) -> float:
                if iter_idx <= t0:
                    p_anneal = 0.0
                elif iter_idx >= t1:
                    p_anneal = 1.0
                else:
                    p_anneal = (iter_idx - t0) / max(1.0, t1 - t0)
                eff_mult = 1.0 + (m - 1.0) * (1.0 - p_anneal)
                return base(iter_idx) * eff_mult / m

            lambdas.append(_hires_lambda)
        else:
            lambdas.append(base_lambda)
    if any(l is not base_lambda for l in lambdas):
        lr_lambda = lambdas  # type: ignore[assignment]

    return LambdaLR(optimizer, lr_lambda)
