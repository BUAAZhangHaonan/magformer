# -*- coding: utf-8 -*-
"""Set criterion with Mask2Former-compatible semantics.

支持分布式训练的损失计算。
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .matcher import HungarianMatcher
from .contrastive_loss import EQOContrastiveLoss


def is_dist_avail_and_initialized() -> bool:
    """检查分布式是否可用且已初始化。"""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """获取分布式世界大小。"""
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def _dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def _sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    *,
    balanced: bool = False,
    min_fg_ratio: float = 0.01,
) -> torch.Tensor:
    """
    Sigmoid cross-entropy loss for masks (point-sampled).

    Default (`balanced=False`) matches Mask2Former: mean over points, sum over masks.

    When `balanced=True`, apply a simple per-mask foreground re-weighting to reduce
    extreme fg/bg imbalance for small objects.
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    if not balanced:
        return loss.mean(1).sum() / num_masks

    # targets: (N, P) where P is number of sampled points
    fg_ratio = targets.mean(dim=1, keepdim=True).clamp(
        min=float(min_fg_ratio))  # (N, 1)
    bg_ratio = 1.0 - fg_ratio

    weights = torch.where(
        targets > 0.5,
        bg_ratio / fg_ratio,     # foreground upweight
        torch.ones_like(bg_ratio),  # background weight = 1
    )
    return (loss * weights).mean(1).sum() / num_masks


def point_sample(input_tensor: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    grid = point_coords * 2.0 - 1.0
    grid = grid.unsqueeze(2)
    sampled = F.grid_sample(
        input_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled.squeeze(-1)


def calculate_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    return -torch.abs(logits)


def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> torch.Tensor:
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled,
                              2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords).squeeze(1)
    uncertainties = calculate_uncertainty(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    idx = torch.topk(uncertainties, k=num_uncertain_points, dim=1)[1]
    idx = idx.unsqueeze(-1).expand(-1, -1, 2)
    uncertain_coords = torch.gather(point_coords, dim=1, index=idx)

    if num_random_points > 0:
        random_coords = torch.rand(
            num_boxes, num_random_points, 2, device=coarse_logits.device)
        return torch.cat([uncertain_coords, random_coords], dim=1)
    return uncertain_coords


class SetCriterion(nn.Module):
    """
    Mask2Former 风格的损失计算。

    This project currently supports exactly one foreground class. Multi-class is not implemented.

    支持：
    - 交叉熵分类损失
    - 掩码 BCE 损失（点采样）
    - Dice 损失
    - 深度监督（辅助输出）
    - 分布式训练（num_masks 归一化）
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float,
        losses: Tuple[str, ...] = ("labels", "masks"),
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        balanced_ce: bool = False,
        balanced_ce_min_fg_ratio: float = 0.01,
        contrastive_enabled: bool = True,
        contrastive_weight: float = 0.5,
        contrastive_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "This project currently supports exactly one foreground class. Multi-class is not implemented. "
                f"Set num_classes=1 (got {num_classes})."
            )
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = tuple(
            loss_name
            for loss_name in losses
            if not (
                loss_name == "depth_boundary"
                and float(weight_dict.get("loss_depth_boundary", 1.0)) == 0.0
            )
        )
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.balanced_ce = bool(balanced_ce)
        self.balanced_ce_min_fg_ratio = float(balanced_ce_min_fg_ratio)

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        # EQO contrastive loss is opt-in/out through runtime config wiring.
        self.contrastive_loss_fn = EQOContrastiveLoss(
            temperature=float(contrastive_temperature)
        )
        self.contrastive_weight = float(contrastive_weight)
        self.contrastive_enabled = bool(contrastive_enabled)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        计算损失。

        Args:
            outputs: 模型输出，包含 pred_logits 和 pred_masks
            targets: 目标列表

        Returns:
            损失字典
        """
        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        # 计算掩码数量，用于归一化
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks_tensor = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(
                iter(outputs.values())).device
        )

        # 分布式训练：同步 num_masks
        if is_dist_avail_and_initialized():
            dist.all_reduce(num_masks_tensor)
        num_masks = torch.clamp(
            num_masks_tensor / get_world_size(), min=1).item()

        losses = {}
        for loss_name in self.losses:
            losses.update(self._get_loss(
                loss_name, outputs, targets, indices, num_masks))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss_name in self.losses:
                    aux_dict = self._get_loss(
                        loss_name, aux_outputs, targets, aux_indices, num_masks)
                    losses.update({f"{k}_{i}": v for k, v in aux_dict.items()})

        total = None
        for key, value in losses.items():
            weight = self.weight_dict.get(key, 1.0)
            weighted = value * weight
            total = weighted if total is None else total + weighted
        losses["total_loss"] = total if total is not None else torch.tensor(
            0.0, device=outputs["pred_logits"].device)
        # EQO Contrastive Loss (patched in)
        if self.contrastive_enabled and 'query_embeddings' in outputs:
            c_loss = self.contrastive_loss_fn(
                outputs['query_embeddings'], targets, indices
            )
            losses['contrastive'] = c_loss * self.contrastive_weight
            # Add to total_loss so gradients flow
            losses['total_loss'] = losses['total_loss'] + losses['contrastive']

        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _loss_labels(self, outputs, targets, indices, num_masks):
        """计算分类损失（交叉熵）。"""
        src_logits = outputs["pred_logits"].float()
        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        if len(indices) > 0 and sum(len(j) for _, j in indices) > 0:
            target_classes_o = torch.cat(
                [t["labels"][j] for t, (_, j) in zip(targets, indices)])
            target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(
            1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def _loss_masks(self, outputs, targets, indices, num_masks):
        """计算掩码损失（BCE + Dice）。"""
        src_masks = outputs["pred_masks"]
        src_idx = self._get_src_permutation_idx(indices)
        if src_idx[0].numel() == 0:
            loss_mask = torch.tensor(0.0, device=src_masks.device)
            loss_dice = torch.tensor(0.0, device=src_masks.device)
            return {"loss_mask": loss_mask, "loss_dice": loss_dice}

        src_masks = src_masks[src_idx]

        target_masks_list = []
        for b, (_, tgt_ids) in enumerate(indices):
            if tgt_ids.numel() == 0:
                continue
            target_masks_list.append(targets[b]["masks"][tgt_ids].float())

        target_masks = torch.cat(target_masks_list, dim=0).to(src_masks)

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                num_points=self.num_points,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
            )
            point_labels = point_sample(target_masks, point_coords).squeeze(1)

        point_logits = point_sample(src_masks, point_coords).squeeze(1)

        loss_mask = _sigmoid_ce_loss(
            point_logits,
            point_labels,
            num_masks,
            balanced=self.balanced_ce,
            min_fg_ratio=self.balanced_ce_min_fg_ratio,
        )
        loss_dice = _dice_loss(point_logits, point_labels, num_masks)

        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    @staticmethod
    def _finite_difference_boundary(inputs: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(inputs[..., :, 1:] - inputs[..., :, :-1])
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = torch.abs(inputs[..., 1:, :] - inputs[..., :-1, :])
        dy = F.pad(dy, (0, 0, 0, 1))
        return dx + dy

    @staticmethod
    def _normalize_depth_for_edges(depths: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        flat_depths = depths.flatten(1)
        flat_valid = valid.flatten(1)
        valid_any = flat_valid.any(dim=1)
        depth_min = flat_depths.masked_fill(~flat_valid, float("inf")).amin(dim=1)
        depth_max = flat_depths.masked_fill(~flat_valid, float("-inf")).amax(dim=1)
        depth_min = torch.where(valid_any, depth_min, torch.zeros_like(depth_min))
        depth_max = torch.where(valid_any, depth_max, torch.ones_like(depth_max))
        denom = (depth_max - depth_min).clamp_min(1e-6).view(-1, 1, 1, 1)
        depth_min = depth_min.view(-1, 1, 1, 1)
        normalized = (depths - depth_min) / denom
        return torch.where(valid, normalized, torch.zeros_like(normalized))

    @staticmethod
    def _depth_boundary_size(height: int, width: int, max_side: int = 256) -> tuple[int, int]:
        if height <= 0 or width <= 0:
            raise ValueError("depth boundary loss requires positive mask spatial dimensions")
        scale = min(1.0, float(max_side) / float(max(height, width)))
        return max(1, int(round(height * scale))), max(1, int(round(width * scale)))

    def _loss_depth_boundary(self, outputs, targets, indices, num_masks):
        del num_masks
        all_masks = outputs["pred_masks"]
        src_idx = self._get_src_permutation_idx(indices)
        if src_idx[0].numel() == 0:
            return {"loss_depth_boundary": all_masks.sum() * 0.0}

        loss_sum = all_masks.sum() * 0.0
        denom_sum = all_masks.new_tensor(0.0)
        for batch_idx, (src_ids, tgt_ids) in enumerate(indices):
            if tgt_ids.numel() == 0:
                continue
            if "depth" not in targets[batch_idx]:
                raise ValueError(
                    "loss_depth_boundary requires target depth for every matched image"
                )
            src_masks = all_masks[batch_idx, src_ids].float()
            height, width = self._depth_boundary_size(
                int(src_masks.shape[-2]), int(src_masks.shape[-1])
            )
            boundary_size = (height, width)
            pred_probs = src_masks.sigmoid().unsqueeze(1)
            if tuple(pred_probs.shape[-2:]) != boundary_size:
                pred_probs = F.interpolate(
                    pred_probs,
                    size=boundary_size,
                    mode="bilinear",
                    align_corners=False,
                )
            pred_boundary = self._finite_difference_boundary(pred_probs)

            depth = targets[batch_idx]["depth"].to(
                device=src_masks.device, dtype=src_masks.dtype
            )
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            if depth.ndim != 3 or depth.shape[0] != 1:
                raise ValueError(
                    "loss_depth_boundary expects each target depth to have shape (1, H, W) or (H, W)"
                )
            valid = depth > 0
            if tuple(depth.shape[-2:]) != boundary_size:
                depth = F.interpolate(
                    depth.unsqueeze(0),
                    size=boundary_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                valid = F.interpolate(
                    valid.float().unsqueeze(0),
                    size=boundary_size,
                    mode="nearest",
                ).squeeze(0).bool()

            if not valid.any():
                continue

            depth_batch = depth.unsqueeze(0).detach()
            valid_batch = valid.unsqueeze(0)
            depth_norm = self._normalize_depth_for_edges(depth_batch, valid_batch)
            depth_edge = self._finite_difference_boundary(depth_norm).detach()
            edge_flat = depth_edge.masked_fill(~valid_batch, 0.0).flatten(1)
            edge_max = edge_flat.amax(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
            depth_edge_norm = (depth_edge / edge_max).clamp(0.0, 1.0).detach()

            valid_float = valid_batch.to(pred_boundary.dtype)
            penalty = pred_boundary * (1.0 - depth_edge_norm) * valid_float
            loss_sum = loss_sum + penalty.sum()
            denom_sum = denom_sum + valid_float.sum().clamp_min(1.0) * pred_boundary.shape[0]

        if denom_sum.item() == 0.0:
            return {"loss_depth_boundary": all_masks.sum() * 0.0}
        return {"loss_depth_boundary": loss_sum / denom_sum.clamp_min(1.0)}

    def _get_loss(self, loss_name, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self._loss_labels,
            "masks": self._loss_masks,
            "depth_boundary": self._loss_depth_boundary,
        }
        return loss_map[loss_name](outputs, targets, indices, num_masks)
