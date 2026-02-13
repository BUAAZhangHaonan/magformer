# -*- coding: utf-8 -*-
"""
YOLOv8-seg Style Visualization

提供类似 YOLOv8 分割结果的可视化功能。
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


# 预定义颜色调色板 (类似 YOLOv8)
COLORS = [
    (255, 0, 0),      # 红
    (0, 255, 0),      # 绿
    (0, 0, 255),      # 蓝
    (255, 255, 0),    # 青
    (255, 0, 255),    # 洋红
    (0, 255, 255),    # 黄
    (128, 0, 255),    # 橙
    (255, 128, 0),    # 天蓝
    (128, 255, 0),    # 春绿
    (255, 0, 128),    # 粉红
    (0, 128, 255),    # 浅蓝
    (128, 0, 128),    # 紫
    (128, 128, 0),    # 橄榄
    (0, 128, 128),    # 青绿
    (255, 128, 128),  # 浅红
    (128, 255, 128),  # 浅绿
    (128, 128, 255),  # 浅蓝
    (64, 0, 128),     # 深紫
    (64, 128, 0),     # 深绿
    (0, 64, 128),     # 深蓝
]


def get_color(idx: int) -> Tuple[int, int, int]:
    """获取索引对应的颜色"""
    return COLORS[idx % len(COLORS)]


def draw_yolov8_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    绘制半透明掩码 (YOLOv8 风格)

    Args:
        image: RGB 图像 (H, W, 3)
        mask: 二值掩码 (H, W)
        color: 颜色 (R, G, B)
        alpha: 透明度

    Returns:
        绘制后的图像
    """
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color

    # 混合原图和掩码
    result = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return result


def draw_yolov8_contour(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """
    绘制掩码轮廓 (YOLOv8 风格)

    Args:
        image: RGB 图像 (H, W, 3)
        mask: 二值掩码 (H, W)
        color: 轮廓颜色 (R, G, B)
        thickness: 线宽

    Returns:
        绘制后的图像
    """
    # 确保掩码是 uint8 类型
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255
    elif mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓 (BGR 格式)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.drawContours(image_bgr, contours, -1, color[::-1], thickness)  # RGB -> BGR

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def draw_yolov8_label(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """
    绘制标签 (YOLOv8 风格，带背景框)

    Args:
        image: RGB 图像 (H, W, 3)
        text: 标签文本
        position: 文本位置 (x, y)
        color: 背景色 (R, G, B)
        font_scale: 字体大小
        thickness: 字体线宽

    Returns:
        绘制后的图像
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 获取文本大小
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 绘制背景框
    x, y = position
    padding = 4
    cv2.rectangle(
        image_bgr,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + padding),
        color[::-1],  # RGB -> BGR
        -1,  # 填充
    )

    # 绘制文本 (白色)
    cv2.putText(
        image_bgr,
        text,
        (x, y - 2),
        font,
        font_scale,
        (255, 255, 255),  # 白色
        thickness,
        cv2.LINE_AA,
    )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def visualize_predictions(
    image: np.ndarray,
    masks: List[np.ndarray],
    scores: List[float],
    labels: Optional[List[int]] = None,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.5,
    alpha: float = 0.4,
    show_labels: bool = True,
    show_contours: bool = True,
    show_masks: bool = True,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    YOLOv8-seg 风格可视化

    Args:
        image: RGB 图像 (H, W, 3) 或 (H, W) 灰度图
        masks: 掩码列表，每个为 (H, W)
        scores: 置信度列表
        labels: 类别标签列表
        class_names: 类别名称列表
        score_threshold: 置信度阈值
        alpha: 掩码透明度
        show_labels: 是否显示标签
        show_contours: 是否显示轮廓
        show_masks: 是否显示掩码
        output_path: 输出路径 (可选)

    Returns:
        可视化后的图像
    """
    # 确保图像是 RGB 格式
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    # 归一化图像到 0-255
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    canvas = image.copy()

    # 过滤低置信度预测
    valid_indices = [i for i, s in enumerate(scores) if s >= score_threshold]

    for idx, i in enumerate(valid_indices):
        mask = masks[i]
        score = scores[i]
        label = labels[i] if labels is not None else 0

        # 确保掩码格式正确
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8) * 255

        # 跳过空掩码
        if mask.sum() == 0:
            continue

        # 获取颜色
        color = get_color(label)

        # 绘制掩码
        if show_masks:
            canvas = draw_yolov8_mask(canvas, mask, color, alpha)

        # 绘制轮廓
        if show_contours:
            canvas = draw_yolov8_contour(canvas, mask, color, thickness=2)

        # 绘制标签
        if show_labels:
            # 获取标签位置 (掩码左上角)
            ys, xs = np.where(mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                x, y = int(xs.min()), int(ys.min())

                # 构建标签文本
                if class_names and label < len(class_names):
                    class_name = class_names[label]
                else:
                    class_name = f"C{label}"
                text = f"{class_name} {score:.2f}"

                canvas = draw_yolov8_label(canvas, text, (x, y), color)

    # 保存结果
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 转换为 BGR 保存
        cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    return canvas


def visualize_batch(
    images: np.ndarray,
    predictions: List[Dict[str, Any]],
    score_threshold: float = 0.5,
    output_dir: Optional[str] = None,
    prefix: str = "vis",
) -> List[np.ndarray]:
    """
    批量可视化

    Args:
        images: 图像批 (N, H, W, 3)
        predictions: 预测结果列表
        score_threshold: 置信度阈值
        output_dir: 输出目录
        prefix: 文件名前缀

    Returns:
        可视化结果列表
    """
    results = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, pred in enumerate(predictions):
        image = images[i] if images.ndim == 4 else images

        masks = pred.get("masks", [])
        scores = pred.get("scores", [])
        labels = pred.get("category_ids", pred.get("labels", None))

        # 处理 labels
        if labels is not None and hasattr(labels, 'cpu'):
            labels = labels.cpu().tolist()
        if labels is None:
            labels = [0] * len(masks)

        output_path = None
        if output_dir:
            output_path = str(output_dir / f"{prefix}_{i:04d}.png")

        vis = visualize_predictions(
            image=image,
            masks=masks,
            scores=scores,
            labels=labels,
            score_threshold=score_threshold,
            output_path=output_path,
        )
        results.append(vis)

    return results


def create_comparison_grid(
    images: List[np.ndarray],
    grid_size: Optional[Tuple[int, int]] = None,
    cell_size: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    创建对比网格图

    Args:
        images: 图像列表
        grid_size: 网格大小 (rows, cols)，如果为 None 则自动计算
        cell_size: 单元格大小 (width, height)

    Returns:
        网格图像
    """
    n = len(images)
    if n == 0:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)

    # 计算网格大小
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size

    # 创建网格画布
    grid = np.zeros((rows * cell_size[1], cols * cell_size[0], 3), dtype=np.uint8)

    for i, img in enumerate(images):
        if i >= rows * cols:
            break

        row = i // cols
        col = i % cols

        # 调整图像大小
        resized = cv2.resize(img, cell_size, interpolation=cv2.INTER_LINEAR)

        # 放置到网格中
        y1 = row * cell_size[1]
        y2 = y1 + cell_size[1]
        x1 = col * cell_size[0]
        x2 = x1 + cell_size[0]

        grid[y1:y2, x1:x2] = resized

    return grid
