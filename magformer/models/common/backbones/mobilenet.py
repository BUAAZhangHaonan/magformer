# -*- coding: utf-8 -*-
"""
MobileNetV3 depth backbone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .depth_base import TimmDepthBackboneBase


class MobileNetV3Depth(TimmDepthBackboneBase):
    """
    MobileNetV3 depth backbone with the standard depth-backbone interface.
    """

    def __init__(
        self,
        variant: str = "small",
        out_features: List[str] = ["res2", "res3", "res4", "res5"],
        pretrained: bool = False,
        weights_path: Optional[str] = None,
    ) -> None:
        self.variant = str(variant).lower()
        super().__init__(
            model_name=self._get_model_name(self.variant),
            out_features=out_features,
            pretrained=pretrained,
            weights_path=weights_path,
        )

    @staticmethod
    def _get_model_name(variant: str) -> str:
        normalized = str(variant).replace("-", "_").lower()
        if normalized in {"small", "mobilenetv3_small", "mobilenetv3_small_100"}:
            return "mobilenetv3_small_100"
        if normalized in {"large", "mobilenetv3_large", "mobilenetv3_large_100"}:
            return "mobilenetv3_large_100"
        raise ValueError(f"Unsupported MobileNetV3 depth variant: {variant}")


def build_mobilenetv3_depth(config: Dict[str, Any]) -> MobileNetV3Depth:
    return MobileNetV3Depth(
        variant=str(config.get("variant", "small")),
        out_features=config.get("out_features", ["res2", "res3", "res4", "res5"]),
        pretrained=bool(config.get("pretrained", False)),
        weights_path=config.get("weights", None),
    )
