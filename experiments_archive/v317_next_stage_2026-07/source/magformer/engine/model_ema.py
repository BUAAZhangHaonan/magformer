"""
Exponential Moving Average (EMA) for model parameters.
Improves evaluation stability by averaging weights over training trajectory.

Usage in trainer:
    ema = ModelEMA(model, decay=0.9999, warmup_iters=200)
    # After optimizer.step():
    ema.update(cur_iter)
    # Before eval:
    ema.apply_shadow()
    # After eval:
    ema.restore()
"""

import copy
import math
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights that is updated as an
    exponential moving average of the training weights. At eval time,
    the shadow weights are swapped in for more stable predictions.

    Args:
        model: The training model (can be DDP-wrapped).
        decay: EMA decay factor. Higher = longer memory. 0.9999 gives
               an effective window of ~10k steps.
        warmup_iters: Number of iterations over which to linearly ramp
                      decay from 0 to the target value. Prevents shadow
                      weights from being stuck at initialization.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_iters: int = 200,
    ):
        self.decay = decay
        self.warmup_iters = warmup_iters

        # Work with the underlying module if DDP-wrapped
        module = model.module if hasattr(model, "module") else model

        # Create shadow parameters as a deepcopy
        self.shadow = {}
        for name, param in module.state_dict().items():
            if param.is_floating_point():
                self.shadow[name] = param.clone().detach()
            else:
                # Non-float params (batch norm counters) share reference
                self.shadow[name] = param.clone()

        self.backup = {}

    def get_decay(self, cur_iter: int) -> float:
        """Compute decay with linear warmup."""
        if cur_iter < self.warmup_iters:
            # Linear ramp from 0 to self.decay
            return self.decay * (cur_iter / self.warmup_iters)
        return self.decay

    @torch.no_grad()
    def update(self, cur_iter: int, model: Optional[nn.Module] = None):
        """Update shadow parameters with current model weights.

        Args:
            cur_iter: Current training iteration (for warmup schedule).
            model: Override model. If None, uses the model passed at init.
                   Must be provided if you don't store a reference.
        """
        decay = self.get_decay(cur_iter)

        module = model.module if hasattr(model, "module") else model
        if module is None:
            raise ValueError("No model provided for EMA update")

        model_state = module.state_dict()
        # foreach form: the per-param mul_/add_ loop issued ~2k tiny kernels
        # per optimizer step (~10-30ms); the fused multi-tensor ops collapse
        # that to a handful of launches with identical numerics.
        shadow_tensors = []
        param_tensors = []
        for name, param in model_state.items():
            shadow = self.shadow.get(name)
            if shadow is not None and param.is_floating_point():
                shadow_tensors.append(shadow)
                param_tensors.append(param.to(shadow.device))
        if shadow_tensors:
            torch._foreach_mul_(shadow_tensors, decay)
            torch._foreach_add_(shadow_tensors, param_tensors, alpha=1.0 - decay)

    def apply_shadow(self, model: nn.Module):
        """Swap in shadow parameters for evaluation.

        Backs up current training weights so they can be restored after eval.

        Only FLOATING-POINT entries are swapped. Non-float entries (integer
        step counters such as fusion's ``_dccg_step`` or the probe step)
        are cloned into the shadow once at init and never updated -- loading
        those clones during eval reset the live schedule state to its
        construction-time value (e.g. DCCG temperature annealing evaluated
        at temp_init for every EMA eval). Leaving them in place lets the
        eval path read the true current schedule state.
        """
        module = model.module if hasattr(model, "module") else model

        # Backup current params (float entries only; nothing else is swapped)
        for name, param in module.state_dict().items():
            if name in self.shadow and param.is_floating_point():
                self.backup[name] = param.clone()

        # Load shadow params
        shadow_state = {
            name: tensor
            for name, tensor in self.shadow.items()
            if tensor.is_floating_point()
        }
        module.load_state_dict(shadow_state, strict=False)
        logger.info("[EMA] Applied shadow parameters for evaluation")

    def restore(self, model: nn.Module):
        """Restore training weights after evaluation."""
        module = model.module if hasattr(model, "module") else model

        # Restore model weights from backup (original training weights)
        model_state = module.state_dict()
        for name in self.backup:
            model_state[name] = self.backup[name]
        module.load_state_dict(model_state, strict=False)

        self.backup = {}
        logger.info("[EMA] Restored training parameters")

    def state_dict(self):
        """Return shadow parameters for checkpoint saving."""
        return {"shadow": self.shadow, "decay": self.decay, "warmup_iters": self.warmup_iters}

    def load_state_dict(self, state_dict):
        """Load shadow parameters from checkpoint."""
        self.shadow = state_dict["shadow"]
        self.decay = state_dict.get("decay", self.decay)
        self.warmup_iters = state_dict.get("warmup_iters", self.warmup_iters)
        logger.info(f"[EMA] Loaded state with decay={self.decay}, warmup_iters={self.warmup_iters}")
