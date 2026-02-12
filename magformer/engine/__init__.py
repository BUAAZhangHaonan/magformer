# -*- coding: utf-8 -*-
"""
MAGFormer Training Engine

纯 PyTorch 实现的训练和评估引擎。
"""

from .trainer import Trainer, DDPTrainer
from .evaluator import COCOEvaluator
from .utils import setup_logger, AverageMeter, ProgressMeter

__all__ = [
    "Trainer",
    "DDPTrainer",
    "COCOEvaluator",
    "setup_logger",
    "AverageMeter",
    "ProgressMeter",
]
