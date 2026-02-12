# -*- coding: utf-8 -*-
"""
Visualizer

统一的可视化类，支持 RGB-D 实例分割结果可视化。
规范: Mask + Box 显示，无左上角 Label 标签。
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

import numpy as np
import cv2
import torch


# =============================================================================
# 颜色生成
# =============================================================================


def generate_colors(num_colors: int, seed: int = 42) -> np.ndarray:
    """
    生成随机颜色列表。

    Args:
        num_colors: 颜色数量
        seed: 随机种子 (保证可复现)

    Returns:
        (N, 3) RGB 颜色数组，范围 [0, 255]
    """
    rng = np.random.RandomState(seed)
    colors = rng.randint(50, 255, size=(num_colors, 3), dtype=np.uint8)
    return colors


def get_palette() -> np.ndarray:
    """
    获取固定调色板 (matplotlib 风格)。

    Returns:
        (N, 3) RGB 颜色数组
    """
    # COCO 标准调色板
    palette = np.array(
        [
            [220, 20, 60],    # 红色
            [119, 11, 32],    # 深红
            [0, 0, 142],       # 深蓝
            [0, 0, 230],       # 黄色
            [106, 0, 228],     # 浅紫
            [0, 60, 100],      # 绿色
            [0, 80, 100],      # 浅绿
            [0, 0, 70],        # 深绿
            [0, 0, 192],       # 浅蓝
            [120, 166, 157],   # 青色
            [0, 100, 100],     # 浅灰
            [0, 0, 128],       # 深灰
            [128, 128, 0],     # 橄榄
            [128, 64, 128],    # 紫色
            [0, 128, 128],     # 蓝绿
            [128, 128, 128],   # 浅灰
            [255, 255, 255],   # 白色
            [192, 192, 192],   # 银色
        ],
        dtype=np.uint8,
    )
    return palette


# =============================================================================
# 掩码和边界框绘制
# =============================================================================


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    在图像上绘制半透明掩码。

    Args:
        image: (H, W, 3) RGB 图像
        mask: (H, W) 二值掩码
        color: RGB 颜色元组
        alpha: 透明度 (0-1)

    Returns:
        绘制后的图像
    """
    result = image.copy()

    # 创建彩色掩码
    colored_mask = np.zeros_like(image)
    colored_mask[mask] = color

    # 混合原图和掩码
    cv2.addWeighted(result, 1 - alpha, colored_mask, alpha, 0, result)

    return result


def draw_bbox(
    image: np.ndarray,
    bbox: List[int],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """
    在图像上绘制边界框。

    Args:
        image: (H, W, 3) RGB 图像
        bbox: [x1, y1, x2, y2] 边界框坐标
        color: RGB 颜色元组
        thickness: 线宽

    Returns:
        绘制后的图像
    """
    result = image.copy()
    x1, y1, x2, y2 = map(int, bbox)

    # 确保坐标在图像范围内
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

    return result


# =============================================================================
# 主可视化类
# =============================================================================


class Visualizer:
    """
    MAGFormer 统一可视化器。

    规范:
    - 叠加半透明 Mask
    - 绘制边界框
    - 不绘制左上角 Label
    """

    def __init__(
        self,
        mask_alpha: float = 0.5,
        bbox_thickness: int = 2,
        random_colors: bool = True,
        color_seed: int = 42,
    ):
        """
        Args:
            mask_alpha: 掩码透明度 (0-1)
            bbox_thickness: 边界框线宽
            random_colors: 是否使用随机颜色
            color_seed: 随机颜色种子
        """
        self.mask_alpha = mask_alpha
        self.bbox_thickness = bbox_thickness
        self.random_colors = random_colors
        self.color_seed = color_seed

        # 预定义调色板
        self.palette = get_palette()

    def draw_instance_predictions(
        self,
        image: np.ndarray,
        masks: Union[np.ndarray, torch.Tensor, List],
        boxes: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
        scores: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
    ) -> np.ndarray:
        """
        绘制实例分割预测结果。

        Args:
            image: (H, W, 3) RGB 图像
            masks: (N, H, W) 掩码数组或列表
            boxes: (N, 4) 边界框数组 [x1, y1, x2, y2] 或列表
            scores: (N,) 置信度分数 (可选，用于过滤)

        Returns:
            可视化图像 (H, W, 3)
        """
        result = image.copy()

        # 转换为列表格式
        if not isinstance(masks, list):
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            masks = [masks[i] for i in range(len(masks))]

        if boxes is not None and not isinstance(boxes, list):
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            boxes = [boxes[i] for i in range(len(boxes))]

        if scores is not None and not isinstance(scores, list):
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            scores = list(scores)

        # 生成颜色
        num_instances = len(masks)
        if self.random_colors:
            colors = generate_colors(num_instances, self.color_seed)
        else:
            colors = self.palette[:num_instances]

        # 绘制每个实例
        for i, (mask, color) in enumerate(zip(masks, colors)):
            # 过滤低分检测
            if scores is not None and scores[i] < 0.5:
                continue

            # 确保 mask 是二值
            if mask.dtype != bool:
                mask = mask > 0.5

            # 绘制掩码
            result = draw_mask(result, mask, tuple(color), self.mask_alpha)

            # 绘制边界框
            if boxes is not None and i < len(boxes):
                result = draw_bbox(
                    result, boxes[i], tuple(color), self.bbox_thickness
                )

        return result

    def draw_depth_map(
        self,
        depth: np.ndarray,
        colormap: int = cv2.COLORMAP_TURBO,
    ) -> np.ndarray:
        """
        可视化深度图。

        Args:
            depth: (H, W) 深度图
            colormap: OpenCV 颜色映射

        Returns:
            (H, W, 3) RGB 深度可视化
        """
        # 归一化深度到 [0, 255]
        depth_norm = depth.copy()
        valid_mask = depth_norm > 0
        depth_norm = depth_norm - depth_norm[valid_mask].min()
        depth_norm = depth_norm / (depth_norm[valid_mask].max() + 1e-6)
        depth_norm = np.clip(depth_norm * 255, 0, 255).astype(np.uint8)

        # 应用颜色映射
        depth_vis = cv2.applyColorMap(depth_norm, colormap)

        # 无效区域设为黑色
        depth_vis[~valid_mask] = 0

        return depth_vis

    def save_image(
        self,
        image: np.ndarray,
        path: str,
    ) -> None:
        """
        保存图像。

        Args:
            image: (H, W, 3) RGB 图像
            path: 保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 转换 RGB 到 BGR (OpenCV 格式)
        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        cv2.imwrite(str(path), image_bgr)


# =============================================================================
# 批量可视化工具
# =============================================================================


def visualize_batch(
    images: torch.Tensor,
    predictions: List[Dict[str, Any]],
    output_dir: str,
    visualizer: Optional[Visualizer] = None,
    max_images: int = 16,
) -> None:
    """
    批量可视化预测结果。

    Args:
        images: (B, C, H, W) 图像批次
        predictions: 预测结果列表
        output_dir: 输出目录
        visualizer: Visualizer 实例 (可选)
        max_images: 最大可视化数量
    """
    if visualizer is None:
        visualizer = Visualizer()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_vis = min(len(images), max_images)

    for i in range(num_vis):
        # 转换图像
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # 转换 RGB 到 RGB (假设输入是 RGB)
        # (如果是 BGR，需要 cv2.cvtColor)

        # 获取预测
        pred = predictions[i] if i < len(predictions) else {}

        masks = pred.get("masks", None)
        boxes = pred.get("boxes", None)
        scores = pred.get("scores", None)

        # 可视化
        result = visualizer.draw_instance_predictions(img, masks, boxes, scores)

        # 保存
        output_file = output_path / f"result_{i:04d}.jpg"
        visualizer.save_image(result, str(output_file))

    print(f"[Visualizer] Saved {num_vis} images to {output_path}")


# =============================================================================
# 网格可视化 (用于 TensorBoard)
# =============================================================================


def make_grid(
    images: torch.Tensor,
    nrow: int = 4,
    padding: int = 2,
    normalize: bool = False,
) -> torch.Tensor:
    """
    创建图像网格。

    Args:
        images: (B, C, H, W) 图像批次
        nrow: 每行图像数
        padding: 间距像素
        normalize: 是否归一化到 [0, 1]

    Returns:
        (C, H_grid, W_grid) 网格图像
    """
    from torchvision.utils import make_grid as make_grid_torch

    grid = make_grid_torch(images, nrow=nrow, padding=padding, normalize=normalize)
    return grid


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    在图像上叠加热力图。

    Args:
        image: (H, W, 3) RGB 图像
        heatmap: (H, W) 热力图
        alpha: 热力图透明度
        colormap: 颜色映射

    Returns:
        叠加后的图像
    """
    result = image.copy()

    # 归一化热力图
    heatmap_norm = heatmap - heatmap.min()
    if heatmap_norm.max() > 0:
        heatmap_norm = heatmap_norm / heatmap_norm.max()
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

    # 应用颜色映射
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # 混合
    cv2.addWeighted(result, 1 - alpha, heatmap_color, alpha, 0, result)

    return result
