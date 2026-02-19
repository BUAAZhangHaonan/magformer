"""
MeanShiftFormer / MSMFormer (vendored baseline).

This checkout is intentionally pruned to "model code only" to keep the main repo
lean. Upstream `meanshiftformer` imports dataset registration and custom dataset
mappers at import time; those modules are not vendored here.

We keep this `__init__` minimal but sufficient for Detectron2 to discover and
register:
- backbones / pixel decoders / heads
- meta-architectures (MeanShiftMaskFormer, PretrainedMeanShiftMaskFormer)
"""

from __future__ import annotations

from . import modeling  # noqa: F401  (registry side effects)
from .config import add_maskformer2_config, add_meanshiftformer_config
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .meanshiftformer_model import MeanShiftMaskFormer
from .pretrained_meanshiftformer_model import PretrainedMeanShiftMaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

__all__ = [
    "add_maskformer2_config",
    "add_meanshiftformer_config",
    "InstanceSegEvaluator",
    "MeanShiftMaskFormer",
    "PretrainedMeanShiftMaskFormer",
    "SemanticSegmentorWithTTA",
]

