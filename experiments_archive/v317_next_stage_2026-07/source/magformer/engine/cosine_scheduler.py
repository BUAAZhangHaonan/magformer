"""
Cosine annealing LR scheduler for MagFormer fine-tuning.

Why cosine over poly for fine-tuning:
- Poly decay drops LR too fast early (power=0.9 means LR at 50% by ~5% of training)
- Cosine keeps LR high longer, giving more exploration time
- At iter 1500/3000: poly LR≈0.5, cosine LR≈0.5 (same midpoint)
- But at iter 500/3000: poly LR≈0.85, cosine LR≈0.93 (cosine retains 8% more)
- This extra exploration in early-mid training is critical when backbone_multiplier
  is raised from 0.1 to 0.3 — the backbone needs time to adapt

Usage:
    from magformer.engine.cosine_scheduler import build_cosine_lr_scheduler

    scheduler = build_cosine_lr_scheduler(optimizer, cfg)
"""

import math
import torch


def _cosine_lr_lambda(cur_iter, warmup_iters, warmup_factor, max_iter):
    """Cosine annealing with linear warmup.

    Returns a multiplier in [0, 1] applied to base_lr.
    - During warmup (cur_iter < warmup_iters): linear ramp from warmup_factor to 1.0
    - After warmup: cosine decay from 1.0 to 0.0
    """
    if cur_iter < warmup_iters:
        if warmup_iters == 0:
            return 1.0
        return warmup_factor + (1.0 - warmup_factor) * cur_iter / warmup_iters
    # Cosine phase: 0.5 * (1 + cos(pi * progress))
    # progress goes from 0 to 1 over [warmup_iters, max_iter]
    progress = (cur_iter - warmup_iters) / max(1, max_iter - warmup_iters)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_cosine_lr_scheduler(optimizer, cfg):
    """Build a cosine annealing LR scheduler with warmup.

    Handles multiple parameter groups (e.g., backbone with different base_lr).

    Args:
        optimizer: The optimizer (potentially with param groups at different LRs)
        cfg: Config object with solver section containing:
            - max_iter: total training iterations
            - warmup_iters: warmup iteration count
            - warmup_factor: initial LR multiplier during warmup

    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """
    max_iter = cfg.solver.max_iter
    warmup_iters = cfg.solver.warmup_iters
    warmup_factor = cfg.solver.warmup_factor

    lr_lambda = lambda cur_iter: _cosine_lr_lambda(
        cur_iter, warmup_iters, warmup_factor, max_iter
    )

    # HDA+ (arena P3-c winner): hires groups get a differentiated schedule —
    # the multiplier baked into their base lr is annealed hires_mult -> 1
    # over [start_frac, end_frac] of max_iter (main-group schedule untouched
    # bit-for-bit). Off by default (hires_mult_anneal absent/False).
    anneal_on = bool(getattr(cfg.solver, "hires_mult_anneal", False))
    lambdas = []
    for group in optimizer.param_groups:
        m = group.get("hires_mult", 1.0)
        if anneal_on and m and float(m) > 1.0:
            m = float(m)
            s_frac = float(getattr(cfg.solver, "hires_anneal_start_frac", 0.1))
            e_frac = float(getattr(cfg.solver, "hires_anneal_end_frac", 0.5))

            def _hires_lambda(cur_iter, m=m, s_frac=s_frac, e_frac=e_frac):
                base_mult = _cosine_lr_lambda(
                    cur_iter, warmup_iters, warmup_factor, max_iter)
                t0, t1 = s_frac * max_iter, e_frac * max_iter
                if cur_iter <= t0:
                    p_anneal = 0.0
                elif cur_iter >= t1:
                    p_anneal = 1.0
                else:
                    p_anneal = (cur_iter - t0) / max(1.0, t1 - t0)
                eff_mult = 1.0 + (m - 1.0) * (1.0 - p_anneal)
                return base_mult * eff_mult / m

            lambdas.append(_hires_lambda)
        else:
            lambdas.append(lr_lambda)
    if len(lambdas) == len(optimizer.param_groups) and any(
            l is not lr_lambda for l in lambdas):
        lr_lambda = lambdas

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
