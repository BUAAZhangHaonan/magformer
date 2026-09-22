# -*- coding: utf-8 -*-
"""EMA Teacher Wrapper for VC-SUDA semi-supervised training.

Maintains an exponential moving average copy of the student model.
The teacher generates pseudo-labels on unlabeled target data.
Uses cosine EMA momentum ramp-up during warmup.
"""

import copy
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List


class EMATeacherWrapper(nn.Module):
    """
    EMA teacher wrapper around a student model.
    
    The teacher is a deep copy of the student, updated via EMA after each
    student optimization step. During warmup, momentum ramps from 0.99 to
    the target value using a cosine schedule.
    
    Usage:
        teacher = EMATeacherWrapper(student_model, momentum=0.999, warmup_steps=500)
        
        # In training loop:
        with torch.no_grad():
            pseudo_outputs = teacher(images, depths)
        # ... student forward + loss ...
        teacher.update_ema(student_model, global_step)
    """
    
    def __init__(
        self,
        student: nn.Module,
        momentum: float = 0.999,
        warmup_steps: int = 500,
    ):
        """
        Args:
            student: Student model to copy
            momentum: Target EMA decay rate
            warmup_steps: Steps to ramp up momentum from 0.99 to target
        """
        super().__init__()
        # Deep copy student as teacher
        self.teacher = copy.deepcopy(student)
        self.teacher.eval()
        # Freeze all teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.initial_momentum = 0.99
    
    @torch.no_grad()
    def update_ema(self, student: nn.Module, step: int) -> float:
        """
        Update teacher parameters via EMA.
        
        During warmup (step < warmup_steps), momentum ramps from initial to target
        using cosine schedule: m = init + (target - init) * (1 - cos(pi * t / T)) / 2
        
        Args:
            student: Current student model
            step: Current training step (global)
            
        Returns:
            Current momentum value used
        """
        if step < self.warmup_steps:
            # Cosine ramp from initial_momentum to self.momentum
            t = step / self.warmup_steps
            current_momentum = self.initial_momentum + (
                self.momentum - self.initial_momentum
            ) * (1 - math.cos(math.pi * t)) / 2
        else:
            current_momentum = self.momentum
        
        # EMA update: teacher = momentum * teacher + (1 - momentum) * student
        for t_param, s_param in zip(
            self.teacher.parameters(), student.parameters()
        ):
            t_param.data.mul_(current_momentum).add_(
                s_param.data, alpha=1 - current_momentum
            )
        
        # Also update buffers (BatchNorm running stats, etc.)
        for t_buf, s_buf in zip(
            self.teacher.buffers(), student.buffers()
        ):
            if t_buf.data.is_floating_point() and s_buf.data.is_floating_point():
                t_buf.data.mul_(current_momentum).add_(
                    s_buf.data, alpha=1 - current_momentum
                )
            else:
                t_buf.data.copy_(s_buf.data)
        
        return current_momentum
    
    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
        padding_masks: Optional[torch.Tensor] = None,
        depth_noise_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher forward pass (no gradients).
        
        Uses the model's forward_inference_decoder_outputs() to get raw
        pred_logits and pred_masks without any post-processing.
        
        Args:
            images: (B, 3, H, W) RGB images
            depths: (B, 1, H, W) depth maps
            targets: Ignored (teacher doesn't compute losses)
            padding_masks: Optional padding masks
            depth_noise_masks: Optional depth noise masks
            
        Returns:
            Dict with pred_logits (B, Nq, C) and pred_masks (B, Nq, H, W)
        """
        return self.teacher.forward_inference_decoder_outputs(
            images=images,
            depths=depths,
            padding_masks=padding_masks,
            depth_noise_masks=depth_noise_masks,
        )
    
    def state_dict(self, *args, **kwargs):
        """Return teacher state dict with 'teacher.' prefix."""
        return {"teacher." + k: v for k, v in self.teacher.state_dict().items()}
    
    def load_state_dict(self, state_dict, strict=True):
        """Load teacher state dict, stripping 'teacher.' prefix."""
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("teacher."):
                cleaned[k[len("teacher."):]] = v
            else:
                cleaned[k] = v
        self.teacher.load_state_dict(cleaned, strict=strict)
