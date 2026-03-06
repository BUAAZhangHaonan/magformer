from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    if logits.min() >= 0.0 and logits.max() <= 1.0:
        return logits.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _connected_components(mask: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return num, labels, centroids


def _split_foreground_by_seeds(
    fg_mask: np.ndarray,
    seed_labels: np.ndarray,
    seed_centroids: np.ndarray,
    min_area: int,
) -> List[np.ndarray]:
    ys, xs = np.nonzero(fg_mask)
    if ys.size == 0:
        return []

    seed_ids = [seed_id for seed_id in np.unique(seed_labels).tolist() if int(seed_id) > 0]
    if not seed_ids:
        return []
    centroids = np.asarray([seed_centroids[seed_id] for seed_id in seed_ids], dtype=np.float32)
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    dists = ((coords[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    nearest = dists.argmin(axis=1)

    masks: List[np.ndarray] = []
    for idx, _seed_id in enumerate(seed_ids):
        mask = np.zeros_like(fg_mask, dtype=np.uint8)
        chosen = nearest == idx
        mask[ys[chosen], xs[chosen]] = 1
        if int(mask.sum()) >= int(min_area):
            masks.append(mask)
    return masks


def instances_from_boundary_logits(
    *,
    fg_logits: np.ndarray,
    boundary_logits: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 20,
) -> List[np.ndarray]:
    fg = (_sigmoid_np(fg_logits) >= float(threshold)).astype(np.uint8)
    boundary = (_sigmoid_np(boundary_logits) >= float(threshold)).astype(np.uint8)
    interior = (fg & (1 - boundary)).astype(np.uint8)
    if interior.sum() == 0:
        interior = fg

    num, labels, centroids = _connected_components(interior)
    if num <= 2:
        num, labels, centroids = _connected_components(fg)

    masks = _split_foreground_by_seeds(fg, labels, centroids, min_area=min_area)
    if masks:
        return masks

    num, labels, _stats, _centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = []
    for seed_id in range(1, num):
        mask = (labels == seed_id).astype(np.uint8)
        if int(mask.sum()) >= int(min_area):
            out.append(mask)
    return out


def instances_from_distance_logits(
    *,
    fg_logits: np.ndarray,
    distance_logits: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 20,
) -> List[np.ndarray]:
    fg = (_sigmoid_np(fg_logits) >= float(threshold)).astype(np.uint8)
    if fg.sum() == 0:
        return []

    distance = distance_logits.astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(distance, kernel, iterations=1)
    peaks = ((distance == dilated) & (distance > 0)).astype(np.uint8) * fg

    num, labels, centroids = _connected_components(peaks)
    if num <= 2:
        num, labels, centroids = _connected_components(fg)

    masks = _split_foreground_by_seeds(fg, labels, centroids, min_area=min_area)
    if masks:
        return masks

    num, labels, _stats, _centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = []
    for seed_id in range(1, num):
        mask = (labels == seed_id).astype(np.uint8)
        if int(mask.sum()) >= int(min_area):
            out.append(mask)
    return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SimpleUNetInstance(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c4, c4 * 2)
        self.up3 = UpBlock(c4 * 2, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, c1)
        self.fg_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.aux_head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        xb = self.bottleneck(self.pool(x4))
        y3 = self.up3(xb, x4)
        y2 = self.up2(y3, x3)
        y1 = self.up1(y2, x2)
        y0 = self.up0(y1, x1)
        return self.fg_head(y0), self.aux_head(y0)


class NestedUNetInstance(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c3, c3 * 2)
        self.up2 = UpBlock(c3 * 2, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, c1)
        self.skip01 = ConvBlock(c1 + c2, c1)
        self.skip12 = ConvBlock(c2 + c3, c2)
        self.fg_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.aux_head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        xb = self.bottleneck(self.pool(x3))
        y2 = self.up2(xb, x3)
        x2p = self.skip12(torch.cat([x2, F.interpolate(y2, size=x2.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        y1 = self.up1(y2, x2p)
        x1p = self.skip01(torch.cat([x1, F.interpolate(y1, size=x1.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        y0 = self.up0(y1, x1p)
        return self.fg_head(y0), self.aux_head(y0)


def build_instance_model(variant: str, in_channels: int = 3, base_channels: int = 32) -> nn.Module:
    if variant in {"unet_boundary_inst", "unet_distance_inst"}:
        return SimpleUNetInstance(in_channels=in_channels, base_channels=base_channels)
    if variant == "unetpp_boundary_inst":
        return NestedUNetInstance(in_channels=in_channels, base_channels=base_channels)
    raise ValueError(f"Unsupported variant: {variant}")
