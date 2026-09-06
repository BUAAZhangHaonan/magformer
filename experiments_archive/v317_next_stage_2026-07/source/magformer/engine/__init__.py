# -*- coding: utf-8 -*-
"""
MAGFormer Training Engine

纯 PyTorch 实现的训练和评估引擎。
"""

from .trainer import Trainer, DDPTrainer
from .utils import setup_logger, AverageMeter, ProgressMeter

__all__ = [
    "Trainer",
    "DDPTrainer",
    "setup_logger",
    "AverageMeter",
    "ProgressMeter",
]


def __getattr__(name):
    if name == "COCOEvaluator":
        from .evaluator import COCOEvaluator
        return COCOEvaluator
    if name in {"VCSUDATrainer", "VCSUDADDPTrainer"}:
        raise ImportError(
            f"{name} has been archived to .trash/dead_modules/vc_suda_trainer.py. "
            f"VC-SUDA domain adaptation is no longer supported."
        )
    raise AttributeError(name)
