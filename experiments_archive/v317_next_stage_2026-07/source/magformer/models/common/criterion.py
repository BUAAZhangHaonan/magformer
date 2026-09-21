# -*- coding: utf-8 -*-
"""Set criterion with Mask2Former-compatible semantics.

支持分布式训练的损失计算。
"""

from typing import Dict, List, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .matcher import HungarianMatcher
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, masks_to_boxes_cxcywh
from torchvision.ops import sigmoid_focal_loss


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
        bg_ratio / fg_ratio,                     # foreground upweight
        fg_ratio / bg_ratio.clamp(min=1e-6),     # background downweight (M7 fix)
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
    - Scale-adaptive loss weighting
    - DN-DETR denoising loss
    - L1 + GIoU box regression loss
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
        dn_enabled: bool = False,
        dn_loss_weight: float = 1.0,
        scale_adaptive_alpha: float = 0.0,
        small_object_sample_threshold: int = 0,
        dn_contrastive_weight: float = 0.5,
        use_uncertainty_weighting: bool = False,
        focal_alpha: float = 0.25,
        mal_enabled: bool = False,
        mal_beta: float = 2.0,
        mal_warmup_iters: int = 9000,
        mal_scale_in_ce: bool = False,
        bass_enabled: bool = False,
        bass_boundary_ratio: float = 0.30,
        bass_interior_ratio: float = 0.20,
        bass_point_floor: int = 32,
        bass_weight_lambda: float = 3.0,
        bass_band_dice_weight: float = 1.0,
        bass_ramp_iters: int = 2000,
        bass_soft_label: bool = True,
        bass_soft_label_mix: float = 1.0,
        **kwargs,  # absorb dead params from arch.py callers
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
        self.eos_coef = float(eos_coef)
        if self.eos_coef < 0.0:
            raise ValueError(f"eos_coef must be non-negative, got {self.eos_coef}")
        self.losses = losses
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.balanced_ce = bool(balanced_ce)
        self.balanced_ce_min_fg_ratio = float(balanced_ce_min_fg_ratio)
        self.focal_alpha = float(focal_alpha)

        # DN-DETR config
        self.dn_enabled = bool(dn_enabled)
        self.dn_loss_weight = float(dn_loss_weight)
        self.dn_contrastive_weight = float(dn_contrastive_weight)

        # Scale-adaptive loss weighting: small objects get higher loss weight.
        # alpha=0 disables; alpha=0.5-1.0 is typical. Weight = (mean_area / (area+1))^alpha
        self.scale_adaptive_alpha = float(scale_adaptive_alpha)

        self.small_object_sample_threshold = int(small_object_sample_threshold)

        # MAL-CP+ (arena P4-c winner): matched queries regress the matcher's
        # soft-Dice quality q instead of one-hot 1 (DEIM-MAL; QFL form
        # |sigmoid(x)-q|^beta * BCE(x, q), negatives/eos untouched). Warmup
        # blends q with 1 over mal_warmup_iters FORWARD calls (~= /9 optimizer
        # steps under deep supervision). mal_scale_in_ce additionally applies
        # the existing per-GT scale weights to matched rows in the CE term
        # (the _compute_scale_weights output was previously computed but
        # ignored by _loss_labels).
        self.mal_enabled = bool(mal_enabled)
        self.mal_beta = float(mal_beta)
        self.mal_warmup_iters = int(mal_warmup_iters)
        self.mal_scale_in_ce = bool(mal_scale_in_ce)
        self._mal_calls = 0

        # BAS-CL+ (arena P3-a winner): boundary-anchored stratified
        # supervision. Three-segment point allocation (uncertainty / GT band
        # / interior), chain-aligned coverage soft labels (avg-pool = the
        # render chain's area-downsample, CF1-construction aligned), band-
        # weighted BCE with balanced_ce band exemption, band Dice; weights
        # ramp over bass_ramp_iters forward calls. The GT geometry pack is
        # built ONCE per criterion.forward and shared by all 9 supervision
        # layers (cross-layer reuse, arena #2 merge).
        self.bass_enabled = bool(bass_enabled)
        self.bass_boundary_ratio = float(bass_boundary_ratio)
        self.bass_interior_ratio = float(bass_interior_ratio)
        self.bass_point_floor = int(bass_point_floor)
        self.bass_weight_lambda = float(bass_weight_lambda)
        self.bass_band_dice_weight = float(bass_band_dice_weight)
        self.bass_ramp_iters = int(bass_ramp_iters)
        self.bass_soft_label = bool(bass_soft_label)
        self.bass_soft_label_mix = float(bass_soft_label_mix)
        self._bass_calls = 0

        # Uncertainty Weighting (Kendall et al. 2018)
        self.use_uncertainty_weighting = bool(use_uncertainty_weighting)
        if self.use_uncertainty_weighting:
            # 5 task groups: ce, mask, dice, bbox, giou
            # Shared across deep supervision layers (loss_ce_0, loss_ce_1, etc. all use same log_var)
            self.log_vars = nn.Parameter(torch.zeros(5))

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

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

        # DN-DETR: strip DN queries so matcher/losses only see regular queries
        dn_active = self.dn_enabled and outputs.get("dn_enabled", False)
        if dn_active:
            num_regular = outputs.get("dn_num_regular_queries",
                                      outputs_without_aux["pred_logits"].shape[1])
            outputs_without_aux = dict(outputs_without_aux)
            outputs_without_aux["pred_logits"] = outputs_without_aux["pred_logits"][:, :num_regular]
            outputs_without_aux["pred_masks"] = outputs_without_aux["pred_masks"][:, :num_regular]

        qualities = None
        if self.mal_enabled:
            indices, qualities = self.matcher(
                outputs_without_aux, targets, return_quality=True)
        else:
            indices = self.matcher(outputs_without_aux, targets)
        self._mal_calls += 1

        bass_pack = None
        if self.bass_enabled:
            with torch.no_grad():
                bass_pack = self._build_bass_pack(targets)
        self._bass_calls += 1

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
        else:
            # Single-process: the count is CPU-derived (from the target
            # lists), so clamp it directly instead of paying a device
            # round-trip via .item() on every micro-step.
            num_masks = float(max(num_masks, 1))

        # For regular losses, use outputs without DN queries
        regular_outputs = outputs_without_aux if dn_active else outputs
        losses = {}

        # Compute scale-adaptive weights once for this matching
        scale_weights = self._compute_scale_weights(targets, indices)

        for loss_name in self.losses:
            losses.update(self._get_loss(
                loss_name, regular_outputs, targets, indices, num_masks,
                scale_weights=scale_weights, qualities=qualities,
                bass_pack=bass_pack))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # DN-DETR: strip DN queries from aux outputs for matching
                if dn_active:
                    num_reg_aux = aux_outputs.get("dn_num_regular_queries",
                                                   aux_outputs["pred_logits"].shape[1])
                    aux_outputs = dict(aux_outputs)
                    aux_outputs["pred_logits"] = aux_outputs["pred_logits"][:, :num_reg_aux]
                    aux_outputs["pred_masks"] = aux_outputs["pred_masks"][:, :num_reg_aux]
                if self.mal_enabled:
                    aux_indices, aux_qualities = self.matcher(
                        aux_outputs, targets, return_quality=True)
                else:
                    aux_indices = self.matcher(aux_outputs, targets)
                    aux_qualities = None

                # Scale-adaptive weights for this aux layer's matching
                aux_scale_weights = self._compute_scale_weights(targets, aux_indices)

                # Use aux_outputs without DN queries for regular loss
                aux_regular = aux_outputs if dn_active else outputs["aux_outputs"][i]
                for loss_name in self.losses:
                    aux_dict = self._get_loss(
                        loss_name, aux_regular, targets, aux_indices, num_masks,
                        scale_weights=aux_scale_weights, qualities=aux_qualities,
                        bass_pack=bass_pack)
                    losses.update({f"{k}_{i}": v for k, v in aux_dict.items()})

        total = None
        if self.use_uncertainty_weighting:
            # Task group mapping: prefix -> log_var index
            _uw_task_map = {
                "loss_ce": 0, "loss_mask": 1, "loss_dice": 2,
                "loss_bbox": 3, "loss_giou": 4,
            }
            # Track regularizer: add 0.5*log_var once per task group, not per aux layer
            _regularizer_added = set()
            for key, value in losses.items():
                # Skip DN losses and fusion entropy -- keep fixed weights
                if key.startswith("loss_dn_") or key == "loss_entropy":
                    weight = self.weight_dict.get(key, 1.0)
                    weighted = value * weight
                else:
                    # Map aux keys: loss_ce_0 -> loss_ce, loss_mask_3 -> loss_mask
                    base_key = key
                    for prefix in _uw_task_map:
                        if key == prefix or key.startswith(prefix + "_"):
                            base_key = prefix
                            break
                    if base_key in _uw_task_map:
                        idx = _uw_task_map[base_key]
                        # Apply weight_dict scaling first (dice_weight=20, mask_weight=5, etc.)
                        w = self.weight_dict.get(key, 1.0)
                        weighted_raw = value * w
                        # Clamp log_vars to prevent exp(-log_var) -> inf
                        log_var = self.log_vars[idx].clamp(min=-10.0, max=10.0)
                        precision = torch.exp(-log_var)
                        # Regularizer once per task group, not per aux layer
                        if base_key not in _regularizer_added:
                            regularizer = 0.5 * log_var
                            _regularizer_added.add(base_key)
                        else:
                            regularizer = torch.tensor(0.0, device=log_var.device)
                        weighted = 0.5 * precision * weighted_raw + regularizer
                    else:
                        weight = self.weight_dict.get(key, 1.0)
                        weighted = value * weight
                total = weighted if total is None else total + weighted
                # Do NOT overwrite raw loss values — keep original for logging
        else:
            for key, value in losses.items():
                weight = self.weight_dict.get(key, 1.0)
                weighted = value * weight
                total = weighted if total is None else total + weighted
        losses["total_loss"] = total if total is not None else torch.tensor(
            0.0, device=outputs["pred_logits"].device)

        # Log UW learned precisions for monitoring
        if self.use_uncertainty_weighting and self.training:
            for i, name in enumerate(["ce", "mask", "dice", "bbox", "giou"]):
                losses[f"uw_precision_{name}"] = torch.exp(-self.log_vars[i]).detach()
                losses[f"uw_logvar_{name}"] = self.log_vars[i].detach()

        # DN-DETR denoising loss
        if self.dn_enabled and outputs.get("dn_enabled", False):
            dn_meta = outputs.get("dn_meta", None)
            dn_losses = self._compute_dn_loss(outputs, targets, dn_meta)
            losses.update(dn_losses)
            dn_total = None
            for key in ["loss_dn_ce", "loss_dn_mask", "loss_dn_dice", "loss_dn_neg_ce"]:
                if key in dn_losses:
                    w = self.weight_dict.get(key, 1.0)
                    weighted = dn_losses[key] * w
                    dn_total = weighted if dn_total is None else dn_total + weighted
            if dn_total is not None:
                losses["total_loss"] = losses["total_loss"] + dn_total

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

    @torch.no_grad()
    def _compute_scale_weights(
        self,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute per-object loss weights inversely proportional to GT mask area.

        Small objects get higher weights so the optimizer pays more attention
        to them.  Weight = (mean_area / (area + 1))^alpha, clamped to [0.5, 5.0].

        Returns:
            Tensor of shape (total_matched_objects,) with per-object weights,
            or None if scale_adaptive_alpha == 0.
        """
        if self.scale_adaptive_alpha <= 0.0:
            return None

        areas = []
        for b, (_, tgt_ids) in enumerate(indices):
            if tgt_ids.numel() == 0:
                continue
            gt_masks = targets[b]["masks"][tgt_ids].float()  # (n, H, W)
            area = gt_masks.flatten(1).sum(dim=1)  # (n,)
            areas.append(area)

        if not areas:
            return None

        areas = torch.cat(areas, dim=0)  # (N,)
        mean_area = areas.mean()
        weights = (mean_area / (areas + 1.0)).pow(self.scale_adaptive_alpha)
        weights = weights.clamp(min=0.5, max=5.0)
        return weights

    def _loss_labels(self, outputs, targets, indices, num_masks,
                     scale_weights=None, qualities=None, bass_pack=None):
        """分类损失（sigmoid focal loss，Mask2Former 标准）。

        MAL-CP+（arena P4-c 胜出，mal_enabled）：matched query 的前景目标从
        one-hot 1 换为匹配器 soft-Dice 质量 q（QFL 形式，warmup 混合），
        负样本/eos 路径逐位不变——增密后必须配质量目标（DEIM 铁律），
        且 q 与匹配 cost 同源（匹配选的就是它，分数学的就是它）。
        """
        src_logits = outputs["pred_logits"].float()  # (B, Q, num_classes+1)
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        matched = len(indices) > 0 and sum(len(j) for _, j in indices) > 0
        if matched:
            target_classes_o = torch.cat(
                [t["labels"][j] for t, (_, j) in zip(targets, indices)])
            target_classes[idx] = target_classes_o

        # one-hot 然后 slice 掉 eos 通道
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        per_logit_loss = sigmoid_focal_loss(
            src_logits, target_classes_onehot,
            alpha=self.focal_alpha, gamma=2, reduction="none",
        )
        query_weights = torch.ones_like(target_classes, dtype=src_logits.dtype)
        query_weights[target_classes == self.num_classes] = self.eos_coef

        mal_active = (
            self.mal_enabled and matched and qualities is not None
            and sum(q.numel() for q in qualities) > 0
        )
        if mal_active:
            q_vec = torch.cat(qualities).to(src_logits.device)  # (num_matched,)
            # warmup: target blends 1 -> q over mal_warmup_iters forward calls
            ramp = 1.0 - min(
                1.0, self._mal_calls / max(1, self.mal_warmup_iters))
            q_eff = ramp + (1.0 - ramp) * q_vec
            b_idx, qpos = idx[0], idx[1]
            cls_idx = target_classes[b_idx, qpos]  # (num_matched,) fg class
            logits_m = src_logits[b_idx, qpos, cls_idx]
            p_m = logits_m.sigmoid()
            qfl = ((p_m - q_eff).abs().pow(self.mal_beta)
                   * F.binary_cross_entropy_with_logits(logits_m, q_eff, reduction="none"))
            per_logit_loss = per_logit_loss.clone()
            per_logit_loss[b_idx, qpos, cls_idx] = qfl
            if self.mal_scale_in_ce and scale_weights is not None:
                # per-GT scale weights on matched rows (previously computed
                # but unused in CE); flat (num_matched,) in the same
                # image-then-index order as q_vec.
                sw = scale_weights.to(src_logits.device)
                query_weights = query_weights.clone()
                query_weights[b_idx, qpos] = query_weights[b_idx, qpos] * sw

        loss_ce = (
            per_logit_loss * query_weights.unsqueeze(-1)
        ).mean() * src_logits.shape[1]
        return {"loss_ce": loss_ce}

    def _loss_masks(self, outputs, targets, indices, num_masks, scale_weights=None, qualities=None, bass_pack=None):
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

        if self.bass_enabled and bass_pack is not None:
            return self._loss_masks_bass(
                src_masks, target_masks, targets, indices, num_masks,
                scale_weights, bass_pack)

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                num_points=self.num_points,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
            )
            point_labels = point_sample(target_masks, point_coords).squeeze(1)

            # Small-object point resampling: replace random points with GT mask points
            if self.small_object_sample_threshold > 0:
                gt_masks_flat = target_masks.squeeze(1)  # (N, H, W)
                target_areas = gt_masks_flat.flatten(1).sum(dim=1)  # (N,)
                point_coords = self._resample_small_object_points(
                    point_coords, gt_masks_flat, target_areas
                )
                # Re-compute labels since point_coords changed
                point_labels = point_sample(target_masks, point_coords).squeeze(1)

        point_logits = point_sample(src_masks, point_coords).squeeze(1)

        # Scale-adaptive weighting: per-object weight from GT area
        if scale_weights is not None:
            ce_per_obj = F.binary_cross_entropy_with_logits(
                point_logits, point_labels, reduction="none"
            ).mean(1)  # (N,)
            if self.balanced_ce:
                fg_ratio = point_labels.mean(dim=1, keepdim=True).clamp(
                    min=float(self.balanced_ce_min_fg_ratio))
                bg_ratio = 1.0 - fg_ratio
                bal_w = torch.where(
                    point_labels > 0.5,
                    bg_ratio / fg_ratio,
                    torch.ones_like(bg_ratio),
                )
                ce_per_obj = (F.binary_cross_entropy_with_logits(
                    point_logits, point_labels, reduction="none"
                ) * bal_w).mean(1)
            loss_mask = (ce_per_obj * scale_weights).sum() / scale_weights.sum().clamp(min=1.0)

            # _dice_loss returns per-instance dice then sums; redo with weights
            inputs = point_logits.sigmoid().flatten(1)
            tgts = point_labels.flatten(1)
            numerator = 2 * (inputs * tgts).sum(-1)
            denominator = inputs.sum(-1) + tgts.sum(-1)
            per_dice = 1 - (numerator + 1) / (denominator + 1)  # (N,)
            loss_dice = (per_dice * scale_weights).sum() / scale_weights.sum().clamp(min=1.0)
        else:
            loss_mask = _sigmoid_ce_loss(
                point_logits,
                point_labels,
                num_masks,
                balanced=self.balanced_ce,
                min_fg_ratio=self.balanced_ce_min_fg_ratio,
            )
            loss_dice = _dice_loss(point_logits, point_labels, num_masks)

        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    def _loss_masks_bass(self, src_masks, target_masks, targets, indices,
                         num_masks, scale_weights, bass_pack):
        """BAS-CL+ three-segment supervision (arena P3-a winner).

        Segments (per matched pair, budget P = num_points): uncertainty
        (1 - b - i share, existing sampler), GT band (max(floor, P*ratio),
        uniform over band cells with replacement), interior (P*0.20).
        Labels come from the coverage soft map (chain-aligned; optional hard
        mix). Band points carry weight lambda (ramped) and are EXEMPT from
        balanced_ce's fg/bg down-weighting (arena #5's discovery: the 0.01
        clamp down-weights boundary-EXTERIOR points ~100x, licensing soft
        slopes). Extra key loss_dice_band (ramped, folded into value).
        """
        P = self.num_points
        N = src_masks.shape[0]
        device = src_masks.device
        self._bass_pack_w = bass_pack["w"]
        self._bass_pack_h = bass_pack["h"]

        # global GT id per matched pair (targets order = pack order)
        gt_ids_list = []
        consumed = 0
        for b, (_, tgt_ids) in enumerate(indices):
            if tgt_ids.numel() == 0:
                continue
            gt_ids_list.append(tgt_ids + consumed)
            consumed += targets[b]["masks"].shape[0]
        gt_ids = torch.cat(gt_ids_list).to(device)  # (N,)

        n_b = max(self.bass_point_floor, int(P * self.bass_boundary_ratio))
        n_i = int(P * self.bass_interior_ratio)
        n_u = max(1, P - n_b - n_i)

        t = min(1.0, self._bass_calls / max(1, self.bass_ramp_iters))
        lam = 1.0 + (self.bass_weight_lambda - 1.0) * t

        with torch.no_grad():
            band_coords = self._bass_sample_pool(
                bass_pack["band_cells"], bass_pack["band_offsets"], gt_ids, n_b)
            interior_coords = self._bass_sample_pool(
                bass_pack["interior_cells"], bass_pack["interior_offsets"],
                gt_ids, n_i)
            unc_coords = get_uncertain_point_coords_with_randomness(
                src_masks, num_points=n_u,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
            )
            point_coords = torch.cat(
                [unc_coords, band_coords, interior_coords], dim=1)  # (N,P,2)

            soft_maps = bass_pack["soft"][gt_ids]  # (N,1,h,w)
            soft_labels = point_sample(soft_maps, point_coords).squeeze(1)
            if self.bass_soft_label and self.bass_soft_label_mix < 1.0:
                hard_labels = point_sample(target_masks, point_coords).squeeze(1)
                mix = self.bass_soft_label_mix
                point_labels = mix * soft_labels + (1 - mix) * hard_labels
            elif self.bass_soft_label:
                point_labels = soft_labels
            else:
                point_labels = point_sample(
                    target_masks, point_coords).squeeze(1)

        point_logits = point_sample(src_masks, point_coords).squeeze(1)

        # per-point weights: band = lam (ramped), else 1; normalize mean-1
        # per mask to keep the loss scale comparable to the baseline.
        w = torch.ones_like(point_labels)
        w[:, n_u:n_u + n_b] = lam
        w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-6)

        bce = F.binary_cross_entropy_with_logits(
            point_logits, point_labels, reduction="none")
        if self.balanced_ce:
            # balanced fg/bg weighting on NON-band points; band exempt
            bal_w = torch.where(
                point_labels > 0.5,
                (1 - point_labels.mean(dim=1, keepdim=True)).clamp(
                    min=self.balanced_ce_min_fg_ratio)
                / point_labels.mean(dim=1, keepdim=True).clamp(min=1e-6),
                torch.ones_like(point_labels),
            )
            is_band = torch.zeros_like(point_labels, dtype=torch.bool)
            is_band[:, n_u:n_u + n_b] = True
            eff_w = torch.where(is_band, w, w * bal_w)
        else:
            eff_w = w
        per_mask_bce = (bce * eff_w).mean(dim=1)  # (N,)
        per_mask_dice = 1 - (
            (2 * (point_logits.sigmoid() * point_labels).sum(1) + 1)
            / ((point_logits.sigmoid().sum(1) + point_labels.sum(1)) + 1))

        # band Dice on the band subset only (scale-adaptive trimap form)
        band_logits = point_logits[:, n_u:n_u + n_b]
        band_labels = point_labels[:, n_u:n_u + n_b]
        band_dice = 1 - (
            (2 * (band_logits.sigmoid() * band_labels).sum(1) + 1)
            / ((band_logits.sigmoid().sum(1) + band_labels.sum(1)) + 1))

        if scale_weights is not None:
            sw = scale_weights.to(device)
            loss_mask = (per_mask_bce * sw).sum() / sw.sum().clamp(min=1.0)
            loss_dice = (per_mask_dice * sw).sum() / sw.sum().clamp(min=1.0)
            loss_band_dice = (band_dice * sw).sum() / sw.sum().clamp(min=1.0)
        else:
            loss_mask = per_mask_bce.sum() / num_masks
            loss_dice = per_mask_dice.sum() / num_masks
            loss_band_dice = band_dice.sum() / num_masks
        # ramp the band dice contribution into the value (weight_dict 1.0)
        loss_band_dice = loss_band_dice * t * self.bass_band_dice_weight

        return {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "loss_dice_band": loss_band_dice,
        }

    @torch.no_grad()
    def _build_bass_pack(self, targets):
        """BAS-CL+ GT geometry pack — built once per criterion.forward,
        shared by every supervision layer (arena P3-a merge of #2).

        For the full GT set: a 512-style coverage soft map (avg-pool = the
        render chain's area-downsample, CF1-construction aligned), the band
        (coverage in (0.05, 0.95) — scale-adaptive by construction), the
        interior (coverage >= 0.95), and GLOBAL flat pools of band/interior
        cells with per-GT offsets for fully-batched uniform sampling.
        """
        device = targets[0]["masks"].device if len(targets) else None
        masks = torch.cat([t["masks"].float() for t in targets], dim=0)
        N, H, W = masks.shape
        h, w = max(1, H // 2), max(1, W // 2)
        soft = F.interpolate(masks[:, None], size=(h, w), mode="area")  # (N,1,h,w)
        soft = soft.clamp(0.0, 1.0)
        band = (soft[:, 0] > 0.05) & (soft[:, 0] < 0.95)   # (N, h, w)
        interior = soft[:, 0] >= 0.95
        band_flat = band.flatten(1)
        interior_flat = interior.flatten(1)

        # global pools sorted by GT with per-GT offsets (batched sampling)
        band_offsets = torch.zeros(N + 1, dtype=torch.long, device=device)
        interior_offsets = torch.zeros(N + 1, dtype=torch.long, device=device)
        band_rows = []
        interior_rows = []
        for i in range(N):
            bi = band_flat[i].nonzero().squeeze(1)
            ii = interior_flat[i].nonzero().squeeze(1)
            band_rows.append(bi)
            interior_rows.append(ii)
            band_offsets[i + 1] = band_offsets[i] + bi.numel()
            interior_offsets[i + 1] = interior_offsets[i] + ii.numel()
        band_cells = torch.cat(band_rows) if band_rows else torch.zeros(
            0, dtype=torch.long, device=device)
        interior_cells = torch.cat(interior_rows) if interior_rows else torch.zeros(
            0, dtype=torch.long, device=device)

        return {
            "soft": soft,                    # (N, 1, h, w) coverage labels
            "h": h, "w": w,
            "band_cells": band_cells,        # flat cell ids, grouped by GT
            "band_offsets": band_offsets,    # (N+1,)
            "interior_cells": interior_cells,
            "interior_offsets": interior_offsets,
            "gt_index_map": None,            # pairs map via targets order
        }

    def _bass_sample_pool(self, pool_cells, offsets, gt_ids, k):
        """Batched uniform-with-replacement sampling of k cells per pair.

        gt_ids: (N_pairs,) global GT index; returns normalized coords
        (N_pairs, k, 2) in (x, y) order for point_sample.
        """
        device = pool_cells.device
        if pool_cells.numel() == 0:
            # degenerate pool (e.g. perfectly grid-aligned GT): fall back to
            # uniform coords so shapes stay static
            return torch.rand(gt_ids.shape[0], k, 2, device=device)
        starts = offsets[gt_ids]              # (N_pairs,)
        lens = (offsets[gt_ids + 1] - starts).clamp(min=1)
        r = torch.rand(gt_ids.shape[0], k, device=device)
        idx = (r * (lens[:, None] - 1).clamp(min=0)).long()
        pick = (starts[:, None] + idx).clamp(
            max=(starts + lens - 1)[:, None])
        rows = pick.clamp(max=pool_cells.numel() - 1 if pool_cells.numel() else 0)
        cells = pool_cells[rows.reshape(-1)].reshape(gt_ids.shape[0], k)
        w_ = self._bass_pack_w
        x = (cells % w_).float() / max(1, w_ - 1)
        y = (cells // w_).float() / max(1, self._bass_pack_h - 1)
        return torch.stack([x, y], dim=-1)

    @torch.no_grad()
    def _resample_small_object_points(self, point_coords, gt_masks, areas):
        """For small objects (area < threshold), replace random points with points
        concentrated inside the GT mask bounding box. This ensures small objects
        get adequate foreground supervision signal.

        Args:
            point_coords: (N, num_points, 2) normalized [0,1] coordinates
            gt_masks: (N, H, W) ground truth masks at feature map resolution
            areas: (N,) GT mask pixel areas

        Returns:
            Modified point_coords with better coverage for small objects
        """
        if self.small_object_sample_threshold <= 0:
            return point_coords

        num_points = point_coords.shape[1]
        num_random = num_points - int(num_points * self.importance_sample_ratio)
        device = point_coords.device

        for i in range(len(areas)):
            if areas[i] >= self.small_object_sample_threshold or areas[i] <= 0:
                continue

            mask = gt_masks[i]  # (H, W)
            H, W = mask.shape

            # Find GT mask pixels
            nonzero = mask.nonzero()  # (K, 2) - rows=y, cols=x
            if len(nonzero) == 0:
                continue

            # Get bounding box with 30% margin
            y_coords = nonzero[:, 0].float()
            x_coords = nonzero[:, 1].float()
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            # Add margin proportional to object size (at least 3px each side)
            margin_y = max(3.0, (0.3 * (y_max - y_min + 1)).item())
            margin_x = max(3.0, (0.3 * (x_max - x_min + 1)).item())
            y_min = torch.clamp(y_min - margin_y, min=0)
            y_max = torch.clamp(y_max + margin_y, max=H - 1)
            x_min = torch.clamp(x_min - margin_x, min=0)
            x_max = torch.clamp(x_max + margin_x, max=W - 1)

            # Sample 70% of random points inside the mask directly, 30% in bbox
            n_mask = int(0.7 * num_random)
            n_bbox = num_random - n_mask

            # Points inside the mask
            if len(nonzero) > 0:
                indices = torch.randint(0, len(nonzero), (n_mask,), device=device)
                mask_points = nonzero[indices].float()  # (n_mask, 2)
                mask_y = mask_points[:, 0] / (H - 1)
                mask_x = mask_points[:, 1] / (W - 1)
            else:
                n_bbox += n_mask
                mask_y = torch.tensor([], device=device)
                mask_x = torch.tensor([], device=device)

            # Points inside bounding box
            bbox_y = torch.rand(n_bbox, device=device) * (y_max - y_min) + y_min
            bbox_x = torch.rand(n_bbox, device=device) * (x_max - x_min) + x_min
            bbox_y_norm = bbox_y / (H - 1)
            bbox_x_norm = bbox_x / (W - 1)

            # Combine and replace the random portion
            new_y = torch.cat([mask_y, bbox_y_norm]) if len(mask_y) > 0 else bbox_y_norm
            new_x = torch.cat([mask_x, bbox_x_norm]) if len(mask_x) > 0 else bbox_x_norm

            # point_sample expects (x, y) order
            new_coords = torch.stack([new_x, new_y], dim=-1)  # (num_random, 2)

            # Replace only the random portion (last num_random points)
            point_coords[i, -num_random:] = new_coords

        return point_coords

    def _compute_dn_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        dn_meta,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for DINO-style contrastive denoising queries.

        Positive queries: reconstruction loss (CE + mask BCE + dice)
        Negative queries: CE loss pushing toward "no object" class
        """
        if dn_meta is None:
            return {}

        src_logits = outputs["pred_logits"]  # (B, num_queries_total, num_classes+1)
        src_masks = outputs["pred_masks"]  # (B, num_queries_total, H, W)

        num_regular = outputs.get("dn_num_regular_queries", src_logits.shape[1])
        B = src_logits.shape[0]

        # Extract DN predictions
        dn_logits = src_logits[:, num_regular:, :]  # (B, num_dn, num_classes+1)
        dn_masks = src_masks[:, num_regular:, :, :]  # (B, num_dn, H, W)

        num_pos = dn_meta["num_positives"]
        num_neg = dn_meta["num_negatives"]
        dn_scalar = dn_meta["dn_scalar"]
        max_valid = dn_meta["max_valid"]
        batch_num_valid = dn_meta["batch_num_valid"]
        gt_labels_padded = dn_meta["gt_labels_padded"]  # (B, max_valid)

        dn_losses = {}

        # === Positive queries: reconstruction loss ===
        if num_pos > 0:
            self_dn_scalar = dn_scalar  # copies per GT slot
            pos_logits = dn_logits[:, :num_pos, :]  # (B, num_pos, C+1)
            pos_masks = dn_masks[:, :num_pos, :, :]  # (B, num_pos, H, W)

            # Build target classes for positive queries
            # Each GT has dn_scalar copies, so query i belongs to GT i // dn_scalar
            target_classes_pos = torch.full((B, num_pos), self.num_classes,
                                             dtype=torch.long, device=dn_logits.device)

            # Build valid mask for positive queries
            valid_mask_pos = torch.zeros(B, num_pos, dtype=torch.bool, device=dn_logits.device)

            for b in range(B):
                nv = batch_num_valid[b]
                if nv == 0:
                    continue
                # Repeat each GT label dn_scalar times
                labels = gt_labels_padded[b, :nv]  # (nv,)
                target_classes_pos[b, :nv * dn_scalar] = labels.unsqueeze(1).expand(nv, dn_scalar).reshape(-1)
                valid_mask_pos[b, :nv * dn_scalar] = True

            # Classification loss for positives
            if valid_mask_pos.any():
                loss_ce = F.cross_entropy(
                    pos_logits.reshape(-1, self.num_classes + 1),
                    target_classes_pos.reshape(-1),
                    reduction='none'
                )
                valid_flat = valid_mask_pos.reshape(-1).float()
                dn_losses["loss_dn_ce"] = (loss_ce * valid_flat).sum() / valid_flat.sum().clamp(min=1)

            # Mask losses for positives
            # Get GT masks for each positive query
            # For each batch, replicate GT masks dn_scalar times
            gt_masks_all = []
            for b in range(B):
                nv = batch_num_valid[b]
                if nv == 0:
                    gt_masks_all.append(torch.zeros(num_pos, pos_masks.shape[2], pos_masks.shape[3],
                                                      device=dn_logits.device))
                    continue
                # Get GT masks from targets
                gt_masks_b = targets[b]["masks"][:nv]  # (nv, H, W)
                # Repeat each mask dn_scalar times
                gt_masks_expanded = gt_masks_b.unsqueeze(1).expand(nv, dn_scalar, -1, -1).reshape(-1,
                                    pos_masks.shape[2], pos_masks.shape[3])
                # Pad to num_pos
                pad_size = num_pos - gt_masks_expanded.shape[0]
                if pad_size > 0:
                    padding = torch.zeros(pad_size, pos_masks.shape[2], pos_masks.shape[3], device=dn_logits.device)
                    gt_masks_expanded = torch.cat([gt_masks_expanded, padding], dim=0)
                gt_masks_all.append(gt_masks_expanded[:num_pos])

            target_masks_pos = torch.stack(gt_masks_all, dim=0)  # (B, num_pos, H, W)

            if valid_mask_pos.any():
                # Sample points for mask loss.
                # D2 fix (arena p4_a audit): the historical uniform 12544-point
                # sampling gives a small GT only ~0-5 foreground points — the
                # exact disease AIM fixed in the matcher, re-introduced on the
                # DN arm. Positives whose GT area < dn_small_gt_area now use a
                # deterministic R x R grid over the GT bbox dilated by 30%
                # (same construction as the AIM window), computed once per GT
                # and shared across its dn_scalar copies. Reduction becomes
                # per-pair mean (previously per-point mean over equal-size
                # pairs, which is equivalent when all pairs share P).
                num_points = self.num_points
                dn_small_area = int(getattr(self, "dn_small_gt_area", 4096))
                R = 64
                H, W = target_masks_pos.shape[-2:]

                per_pair_bce = []
                per_pair_dice = []
                per_pair_valid = []
                for b in range(B):
                    nv = batch_num_valid[b]
                    if nv == 0:
                        for _ in range(num_pos):
                            per_pair_bce.append(None)
                            per_pair_dice.append(None)
                            per_pair_valid.append(False)
                        continue
                    gt_masks_b = targets[b]["masks"][:nv]  # (nv, H, W)
                    areas = gt_masks_b.flatten(1).sum(dim=1)
                    for g in range(nv):
                        area = float(areas[g])
                        small = area < dn_small_area and area > 0
                        if small:
                            m = gt_masks_b[g] > 0.5
                            any_row = m.any(dim=1).float()
                            any_col = m.any(dim=0).float()
                            y0 = int(any_row.argmax())
                            y1 = int(H - 1 - any_row.flip(0).argmax())
                            x0 = int(any_col.argmax())
                            x1 = int(W - 1 - any_col.flip(0).argmax())
                            my = max(3.0, 0.3 * (y1 - y0 + 1))
                            mx = max(3.0, 0.3 * (x1 - x0 + 1))
                            y0c = max(0.0, y0 - my)
                            y1c = min(float(H - 1), y1 + my)
                            x0c = max(0.0, x0 - mx)
                            x1c = min(float(W - 1), x1 + mx)
                            base = torch.linspace(0.0, 1.0, R, device=dn_logits.device)
                            ys = y0c + base * (y1c - y0c)
                            xs = x0c + base * (x1c - x0c)
                            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                            coords = torch.stack(
                                [gx / (W - 1), gy / (H - 1)], dim=-1
                            ).reshape(1, -1, 2)  # (1, P, 2) normalized (x, y)
                            P = R * R
                        else:
                            coords = torch.rand(1, num_points, 2, device=dn_logits.device)
                            P = num_points
                        # Shared coords across the dn_scalar copies of this GT
                        for s in range(self_dn_scalar):
                            qi = g * self_dn_scalar + s  # pair index within (b, :)
                            tgt_pts = point_sample(
                                gt_masks_b[g][None, None].float(), coords
                            ).squeeze(1).squeeze(0)  # (P,)
                            src_pts = point_sample(
                                pos_masks[b, qi][None, None], coords
                            ).squeeze(1).squeeze(0)  # (P,)
                            bce = F.binary_cross_entropy_with_logits(
                                src_pts, tgt_pts, reduction='mean')
                            src_soft = src_pts.sigmoid()
                            inter = (src_soft * tgt_pts).sum()
                            denom = src_soft.sum() + tgt_pts.sum()
                            dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
                            per_pair_bce.append(bce)
                            per_pair_dice.append(dice)
                            per_pair_valid.append(True)
                    # pad invalid slots (nv..cap) for alignment
                    for _ in range(num_pos - nv * self_dn_scalar):
                        per_pair_bce.append(None)
                        per_pair_dice.append(None)
                        per_pair_valid.append(False)

                valid_flags = torch.tensor(
                    [v for v in per_pair_valid], device=dn_logits.device)
                if valid_flags.any():
                    bce_stack = torch.stack(
                        [x for x, v in zip(per_pair_bce, per_pair_valid) if v])
                    dice_stack = torch.stack(
                        [x for x, v in zip(per_pair_dice, per_pair_valid) if v])
                    dn_losses["loss_dn_mask"] = bce_stack.mean()
                    dn_losses["loss_dn_dice"] = dice_stack.mean()

        # === Negative queries: push toward "no object" ===
        if num_neg > 0:
            neg_logits = dn_logits[:, num_pos:num_pos + num_neg, :]  # (B, num_neg, C+1)

            # Build valid mask for negative queries. D3 fix: validity comes
            # from the generator's IoU-gated negative resampling (slots whose
            # displaced box still overlaps a real GT after 5 tries are
            # invalid), NOT the old "first nv slots" prefix.
            neg_valid = dn_meta.get("neg_valid", None)
            if neg_valid is not None:
                valid_mask_neg = neg_valid.to(device=dn_logits.device, dtype=torch.bool)
            else:
                valid_mask_neg = torch.zeros(B, num_neg, dtype=torch.bool, device=dn_logits.device)
                for b in range(B):
                    nv = batch_num_valid[b]
                    if nv > 0:
                        valid_mask_neg[b, :nv] = True

            # Target: "no object" class = self.num_classes
            target_classes_neg = torch.full((B, num_neg), self.num_classes,
                                             dtype=torch.long, device=dn_logits.device)

            if valid_mask_neg.any():
                loss_neg_ce = F.cross_entropy(
                    neg_logits.reshape(-1, self.num_classes + 1),
                    target_classes_neg.reshape(-1),
                    reduction='none'
                )
                valid_flat = valid_mask_neg.reshape(-1).float()
                dn_losses["loss_dn_neg_ce"] = (loss_neg_ce * valid_flat).sum() / valid_flat.sum().clamp(min=1)

        return dn_losses

    def _loss_boxes(self, outputs, targets, indices, num_masks, scale_weights=None, qualities=None, bass_pack=None):
        """L1 + GIoU box regression loss (MaskDINO-style)."""
        assert "pred_boxes" in outputs, "pred_boxes required for box loss"
        src_boxes = outputs["pred_boxes"]  # (B, Q, 4) cxcywh normalized [0,1]
        src_idx = self._get_src_permutation_idx(indices)
        if src_idx[0].numel() == 0:
            return {"loss_bbox": torch.tensor(0.0, device=src_boxes.device),
                    "loss_giou": torch.tensor(0.0, device=src_boxes.device)}

        src_boxes = src_boxes[src_idx]  # (N, 4) cxcywh normalized

        target_boxes_list = []
        for b, (_, tgt_ids) in enumerate(indices):
            if tgt_ids.numel() == 0:
                continue
            all_target_boxes = masks_to_boxes_cxcywh(targets[b]["masks"].float())
            target_boxes_list.append(all_target_boxes[tgt_ids])

        target_boxes = torch.cat(target_boxes_list, dim=0).to(src_boxes)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum(-1).mean()

        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes),
        ).diagonal()
        loss_giou = 1 - giou.mean()

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def _get_loss(self, loss_name, outputs, targets, indices, num_masks,
                  scale_weights=None, qualities=None, bass_pack=None):
        loss_map = {
            "labels": self._loss_labels,
            "masks": self._loss_masks,
            "boxes": self._loss_boxes,
        }
        return loss_map[loss_name](
            outputs, targets, indices, num_masks,
            scale_weights=scale_weights, qualities=qualities,
            bass_pack=bass_pack)
