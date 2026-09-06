# -*- coding: utf-8 -*-
"""Shared box geometry used by matching and box supervision."""

import torch


def masks_to_boxes_cxcywh(masks: torch.Tensor) -> torch.Tensor:
    """Derive normalized ``cxcywh`` boxes from binary instance masks.

    The right and bottom edges are exclusive, so a one-pixel mask has a
    non-zero width and height. Every instance mask must contain foreground.
    """
    if masks.ndim != 3:
        raise ValueError(
            f"masks must have shape (N, H, W), got {tuple(masks.shape)}"
        )

    num_masks, height, width = masks.shape
    if height <= 0 or width <= 0:
        raise ValueError(
            f"mask spatial dimensions must be positive, got {(height, width)}"
        )

    dtype = masks.dtype if torch.is_floating_point(masks) else torch.float32
    if num_masks == 0:
        return torch.empty((0, 4), dtype=dtype, device=masks.device)

    foreground = masks > 0.5
    has_foreground = foreground.flatten(1).any(dim=1)
    if not bool(has_foreground.all()):
        empty_indices = torch.where(~has_foreground)[0].tolist()
        raise ValueError(
            "instance masks must contain foreground; empty mask indices: "
            f"{empty_indices}"
        )

    occupied_x = foreground.any(dim=1)
    occupied_y = foreground.any(dim=2)
    x_coordinates = torch.arange(width, device=masks.device)
    y_coordinates = torch.arange(height, device=masks.device)

    x_min = torch.where(
        occupied_x,
        x_coordinates.unsqueeze(0),
        torch.full_like(x_coordinates.unsqueeze(0), width),
    ).amin(dim=1)
    y_min = torch.where(
        occupied_y,
        y_coordinates.unsqueeze(0),
        torch.full_like(y_coordinates.unsqueeze(0), height),
    ).amin(dim=1)
    x_max_exclusive = torch.where(
        occupied_x,
        x_coordinates.unsqueeze(0) + 1,
        torch.zeros_like(x_coordinates.unsqueeze(0)),
    ).amax(dim=1)
    y_max_exclusive = torch.where(
        occupied_y,
        y_coordinates.unsqueeze(0) + 1,
        torch.zeros_like(y_coordinates.unsqueeze(0)),
    ).amax(dim=1)

    x_min = x_min.to(dtype=dtype) / float(width)
    y_min = y_min.to(dtype=dtype) / float(height)
    x_max_exclusive = x_max_exclusive.to(dtype=dtype) / float(width)
    y_max_exclusive = y_max_exclusive.to(dtype=dtype) / float(height)

    return torch.stack(
        (
            (x_min + x_max_exclusive) * 0.5,
            (y_min + y_max_exclusive) * 0.5,
            x_max_exclusive - x_min,
            y_max_exclusive - y_min,
        ),
        dim=-1,
    )


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from ``cxcywh`` to ``xyxy`` without changing scale."""
    if boxes.shape[-1] != 4:
        raise ValueError(f"boxes must end in four coordinates, got {tuple(boxes.shape)}")
    center_x, center_y, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise generalized IoU for two sets of ``xyxy`` boxes."""
    if boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError(f"boxes1 must have shape (N, 4), got {tuple(boxes1.shape)}")
    if boxes2.ndim != 2 or boxes2.shape[-1] != 4:
        raise ValueError(f"boxes2 must have shape (M, 4), got {tuple(boxes2.shape)}")

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    intersection_left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    intersection_right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_size = (intersection_right_bottom - intersection_left_top).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]

    union = area1[:, None] + area2[None, :] - intersection
    eps = torch.finfo(union.dtype).eps
    iou = intersection / union.clamp_min(eps)

    enclosing_left_top = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_size = (enclosing_right_bottom - enclosing_left_top).clamp(min=0)
    enclosing_area = enclosing_size[..., 0] * enclosing_size[..., 1]
    return iou - (enclosing_area - union) / enclosing_area.clamp_min(eps)
