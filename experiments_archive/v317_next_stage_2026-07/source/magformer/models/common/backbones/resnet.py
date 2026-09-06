# -*- coding: utf-8 -*-
"""
ResNet depth backbone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .depth_base import TimmDepthBackboneBase


class ResNetDepth(TimmDepthBackboneBase):
    """
    ResNet depth backbone with the standard depth-backbone interface.
    """

    def __init__(
        self,
        depth: int = 18,
        out_features: List[str] = ["res2", "res3", "res4", "res5"],
        pretrained: bool = False,
        weights_path: Optional[str] = None,
    ) -> None:
        self.depth = int(depth)
        super().__init__(
            model_name=self._get_model_name(self.depth),
            out_features=out_features,
            pretrained=pretrained,
            weights_path=weights_path,
        )

    @staticmethod
    def _get_model_name(depth: int) -> str:
        if int(depth) == 18:
            return "resnet18"
        raise ValueError(f"Unsupported ResNet depth backbone depth: {depth}")


def build_resnet_depth(config: Dict[str, Any]) -> ResNetDepth:
    return ResNetDepth(
        depth=int(config.get("depth", 18)),
        out_features=config.get(
            "out_features", ["res2", "res3", "res4", "res5"]),
        pretrained=bool(config.get("pretrained", False)),
        weights_path=config.get("weights", None),
    )
