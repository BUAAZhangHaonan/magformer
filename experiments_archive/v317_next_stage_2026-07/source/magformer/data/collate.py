# -*- coding: utf-8 -*-
"""
Collate Functions for Batch Processing

批处理整理函数，处理变长输入和特殊数据格式。
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
import numpy as np


# =============================================================================
# 实例分割 Collate
# =============================================================================
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    实例分割数据集的批处理整理函数。

    Args:
        batch: 样本列表，每个样本包含:
            - images: (C, H, W) Tensor
            - depths: (1, H, W) 或 (H, W) Tensor
            - annotations: 标注列表 (训练时)
            - image_ids: 图像 ID
            - (可选) noise_masks: (1, H, W) Tensor

    Returns:
        整理后的批次:
            - images: (B, C, H, W) Tensor
            - depths: (B, 1, H, W) Tensor
            - image_ids: (B,) Tensor
            - targets: 目标字典 (训练时)
    """
    # 提取字段
    images = [item["image"] for item in batch]
    depths = [item["depth"] for item in batch]
    image_ids = [item.get("image_id", 0) for item in batch]

    # 堆叠图像和深度
    images = torch.stack(images, dim=0)
    depths = torch.stack(depths, dim=0)

    result = {
        "images": images,
        "depths": depths,
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
    }

    if "depth_valid_mask" in batch[0]:
        depth_valid_masks = [item["depth_valid_mask"].bool() for item in batch]
        result["depth_valid_masks"] = torch.stack(depth_valid_masks, dim=0)

    # content/padding masks (for transformers): True means padding
    if "content_mask" in batch[0]:
        content_masks = [item["content_mask"] for item in batch]
        content_masks = torch.stack(content_masks, dim=0).bool()
        result["content_masks"] = content_masks
        result["padding_masks"] = ~content_masks

    # 噪声掩码
    if "noise_mask" in batch[0]:
        noise_masks = []
        for item in batch:
            nm = item["noise_mask"]
            if isinstance(nm, np.ndarray):
                nm = torch.from_numpy(nm.astype(np.float32))
            if nm.ndim == 2:
                nm = nm.unsqueeze(0)
            noise_masks.append(nm.float())
        result["noise_masks"] = torch.stack(noise_masks, dim=0)

    # 训练时处理标注
    if "masks" in batch[0]:
        targets = _process_instances(batch)
        result["targets"] = targets
    elif "annotations" in batch[0]:
        targets = _process_annotations(batch)
        result["targets"] = targets

    return result


def _process_instances(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """处理 dataset 提供的 masks/boxes/labels。"""
    targets = []
    for item in batch:
        labels = item.get("labels", torch.zeros(0, dtype=torch.long))
        masks = item.get("masks", torch.zeros(
            (0, item["image"].shape[1], item["image"].shape[2]), dtype=torch.bool))
        boxes = item.get("boxes", torch.zeros((0, 4), dtype=torch.float32))

        if isinstance(masks, torch.Tensor) and masks.dtype != torch.bool:
            masks = masks > 0.5

        targets.append({
            "labels": labels,
            "masks": masks,
            "boxes": boxes,
            "image_id": item.get("image_id", 0),
            "orig_size": torch.tensor(item["image"].shape[1:]),
        })

    return targets


def _process_annotations(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    处理 COCO 格式标注。

    Args:
        batch: 包含 annotations 的样本列表

    Returns:
        处理后的目标列表，每个包含:
            - labels: (N,) 类别标签
            - masks: (N, H, W) 二值掩码
            - boxes: (N, 4) 边界框 [x1, y1, x2, y2]
    """
    targets = []

    for item in batch:
        annotations = item["annotations"]
        if not annotations:
            # 空标注
            targets.append({
                "labels": torch.zeros(0, dtype=torch.long),
                "masks": torch.zeros((0, item["image"].shape[1], item["image"].shape[2]), dtype=torch.bool),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
            })
            continue

        # 提取字段
        labels = []
        masks = []
        boxes = []

        for ann in annotations:
            labels.append(ann["category_id"])

            # 处理分割掩码
            if "segmentation" in ann:
                mask = _decode_mask(ann["segmentation"],
                                    item["image"].shape[1:])
                masks.append(mask)

            # 处理边界框 (COCO 格式: [x, y, w, h])
            if "bbox" in ann:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])

        # 转换为 Tensor
        targets.append({
            "labels": torch.tensor(labels, dtype=torch.long),
            "masks": torch.stack(masks) if masks else torch.zeros((0, item["image"].shape[1], item["image"].shape[2]), dtype=torch.bool),
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
        })

    return targets


def _decode_mask(segmentation: Any, img_size: Tuple[int, int]) -> torch.Tensor:
    """
    解码 COCO 分割格式为二值掩码。

    Args:
        segmentation: COCO 分割 (polygon 或 RLE)
        img_size: 图像尺寸 (H, W)

    Returns:
        (H, W) 二值掩码 Tensor
    """
    from pycocotools import mask as coco_mask

    # 使用 pycocotools 解码
    if isinstance(segmentation, list):
        # Polygon 格式
        rles = coco_mask.frPyObjects(segmentation, img_size[0], img_size[1])
    else:
        # RLE 格式
        rles = [segmentation]

    # 解码为掩码
    mask = coco_mask.decode(rles)

    # 多个对象合并
    if len(mask.shape) == 3:
        mask = mask.any(axis=2)

    return torch.from_numpy(mask.astype(bool))


# =============================================================================
# Padding Collate (用于变长输入)
# =============================================================================
def pad_collate_fn(batch: List[Dict[str, Any]], pad_value: float = 0.0) -> Dict[str, Any]:
    """
    带填充的批处理整理函数，用于变长输入。

    Args:
        batch: 样本列表
        pad_value: 填充值

    Returns:
        整理后的批次，带 padding_mask
    """
    # 找到最大尺寸
    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)

    padded_images = []
    padded_depths = []
    padded_depth_valid_masks = []
    padding_masks = []

    for item in batch:
        img = item["image"]
        depth = item["depth"]

        # 计算填充
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]

        # 填充图像
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), value=pad_value)
        padded_images.append(padded_img)

        # 填充深度
        padded_depth = F.pad(depth, (0, pad_w, 0, pad_h), value=pad_value)
        padded_depths.append(padded_depth)
        if "depth_valid_mask" not in item:
            raise KeyError("pad_collate_fn requires depth_valid_mask")
        padded_depth_valid_masks.append(
            F.pad(
                item["depth_valid_mask"].bool(),
                (0, pad_w, 0, pad_h),
                value=False,
            )
        )

        # 创建 padding mask (True 表示填充区域)
        pad_mask = torch.zeros((1, max_h, max_w), dtype=torch.bool)
        if pad_h > 0 or pad_w > 0:
            pad_mask[:, max_h - pad_h:, :] = True
            pad_mask[:, :, max_w - pad_w:] = True
        padding_masks.append(pad_mask)

    return {
        "images": torch.stack(padded_images, dim=0),
        "depths": torch.stack(padded_depths, dim=0),
        "depth_valid_masks": torch.stack(padded_depth_valid_masks, dim=0),
        "padding_masks": torch.stack(padding_masks, dim=0),
        "image_ids": torch.tensor([item.get("image_id", 0) for item in batch]),
    }


# =============================================================================
# 辅助函数
# =============================================================================
def convert_coco_box_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    转换 COCO 边界框格式 [x, y, w, h] 为 [x1, y1, x2, y2]。

    Args:
        boxes: (N, 4) 边界框 [x, y, w, h]

    Returns:
        (N, 4) 边界框 [x1, y1, x2, y2]
    """
    converted = boxes.clone()
    converted[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
    converted[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
    return converted


def clip_boxes_to_image(boxes: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
    """
    裁剪边界框到图像范围内。

    Args:
        boxes: (N, 4) 边界框 [x1, y1, x2, y2]
        img_size: 图像尺寸 (H, W)

    Returns:
        裁剪后的边界框
    """
    h, w = img_size
    clipped = boxes.clone()

    clipped[:, 0] = torch.clamp(clipped[:, 0], min=0, max=w - 1)
    clipped[:, 1] = torch.clamp(clipped[:, 1], min=0, max=h - 1)
    clipped[:, 2] = torch.clamp(clipped[:, 2], min=0, max=w - 1)
    clipped[:, 3] = torch.clamp(clipped[:, 3], min=0, max=h - 1)

    return clipped


def compute_box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    计算边界框面积。

    Args:
        boxes: (N, 4) 边界框 [x1, y1, x2, y2]

    Returns:
        (N,) 面积
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
