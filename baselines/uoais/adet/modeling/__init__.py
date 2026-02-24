# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .backbone import build_fcos_resnet_fpn_backbone
from .rcnn import ORCNNROIHeads

# Optional text head (requires compiled ops via `adet._C`). Keep UOAIS / GeneralizedRCNN
# baselines importable on modern PyTorch even when these ops are unavailable.
try:  # pragma: no cover
    from .roi_heads.text_head import TextHead  # noqa: F401
except Exception:
    pass

_EXCLUDE = {"torch", "ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
