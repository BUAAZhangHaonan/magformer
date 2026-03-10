# -*- coding: utf-8 -*-
"""
ConvNeXt depth backbone.
"""

from typing import Dict, List, Optional, Any

from .depth_base import TimmDepthBackboneBase


class ConvNeXtDepth(TimmDepthBackboneBase):
    """
    ConvNeXt depth backbone with the standard depth-backbone interface.
    """

    def __init__(
        self,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale: float = 1e-6,
        out_features: List[str] = ["res2", "res3", "res4", "res5"],
        pretrained: bool = False,
        weights_path: Optional[str] = None,
    ):
        """
        Args:
            depths: 每层深度
            dims: 每层维度
            drop_path_rate: Drop Path 比率
            layer_scale: Layer Scale 初始值
            out_features: 输出特征层名称
            pretrained: 是否使用预训练权重
            weights_path: 预训练权重路径
        """
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.layer_scale = layer_scale
        super().__init__(
            model_name=self._get_model_name(dims),
            out_features=out_features,
            pretrained=pretrained,
            weights_path=weights_path,
        )

    def _get_model_name(self, dims: List[int]) -> str:
        """根据维度确定模型名称"""
        if dims[0] == 64:
            return "convnext_tiny"
        elif dims[0] == 96:
            return "convnext_tiny"
        elif dims[0] == 128:
            return "convnext_small"
        elif dims[0] == 192:
            return "convnext_base"
        elif dims[0] == 256:
            return "convnext_large"
        else:
            return "convnext_tiny"  # 默认


def build_convnext_depth(
    config: Dict[str, Any]
) -> ConvNeXtDepth:
    """
    根据配置构建 ConvNeXt 深度骨干网络。

    Args:
        config: ConvNeXt 配置字典

    Returns:
        ConvNeXtDepth 模型
    """
    return ConvNeXtDepth(
        depths=config.get("depths", [3, 3, 9, 3]),
        dims=config.get("dims", [96, 192, 384, 768]),
        drop_path_rate=config.get("drop_path_rate", 0.0),
        layer_scale=config.get("layer_scale", 1e-6),
        out_features=config.get(
            "out_features", ["res2", "res3", "res4", "res5"]),
        pretrained=config.get("pretrained", False),
        weights_path=config.get("weights", None),
    )
