# -*- coding: utf-8 -*-
"""VC-SUDA Criterion: extends SetCriterion with pseudo-label losses.

Adds quality-weighted pseudo-label loss on top of the standard supervised
SetCriterion losses. Pseudo-labels come from the EMA teacher and are
filtered by quality score. Hungarian matching aligns student queries to
teacher pseudo-labels before loss computation.
"""

from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from magformer.models.common.criterion import SetCriterion


class VCSUDACriterion(nn.Module):
    """
    Extended criterion for VC-SUDA training.

    Composes:
    1. Standard supervised loss (via SetCriterion) on labeled source data
    2. Pseudo-label loss on unlabeled target data (quality-weighted CE + BCE + Dice)
    3. Domain adaptation losses (added externally by trainer)

    Pseudo-label loss uses Hungarian matching to find the optimal assignment
    between student queries and teacher pseudo-labels, then computes
    quality-weighted losses on the matched pairs.
    """

    def __init__(
        self,
        supervised_criterion: SetCriterion,
        pseudo_weight_ce: float = 2.0,
        pseudo_weight_mask: float = 5.0,
        pseudo_weight_dice: float = 5.0,
        pseudo_no_object_weight: float = 0.1,
        pseudo_unmatched_negative_enabled: bool = False,
        pseudo_unmatched_negative_weight: float = 0.05,
        pseudo_unmatched_negative_score_thresh: float = 0.9,
        pseudo_exterior_ring_enabled: bool = False,
        pseudo_exterior_ring_weight: float = 0.0,
        pseudo_exterior_ring_radius: int = 2,
    ):
        """
        Args:
            supervised_criterion: Standard SetCriterion for supervised loss
            pseudo_weight_ce: Weight for pseudo-label classification loss
            pseudo_weight_mask: Weight for pseudo-label mask BCE loss
            pseudo_weight_dice: Weight for pseudo-label dice loss
            pseudo_no_object_weight: Deprecated compatibility argument; pseudo labels are
                partial positives, so unmatched queries are ignored.
            pseudo_unmatched_negative_enabled: Enable background CE for high-score
                unmatched queries in the pseudo-label branch.
            pseudo_unmatched_negative_weight: Weight for high-score unmatched background CE.
            pseudo_unmatched_negative_score_thresh: Foreground score threshold for
                selecting unmatched queries.
            pseudo_exterior_ring_enabled: Enable matched pseudo-positive exterior
                ring probability penalty in the pseudo-label branch.
            pseudo_exterior_ring_weight: Weight for matched exterior ring penalty.
            pseudo_exterior_ring_radius: Dilation radius for the exterior ring.
        """
        super().__init__()
        if pseudo_unmatched_negative_score_thresh < 0.0 or pseudo_unmatched_negative_score_thresh > 1.0:
            raise ValueError(
                "pseudo_unmatched_negative_score_thresh must be in [0, 1], "
                f"got {pseudo_unmatched_negative_score_thresh}"
            )
        if pseudo_unmatched_negative_weight < 0.0:
            raise ValueError(
                "pseudo_unmatched_negative_weight must be non-negative, "
                f"got {pseudo_unmatched_negative_weight}"
            )
        if pseudo_exterior_ring_weight < 0.0:
            raise ValueError(
                "pseudo_exterior_ring_weight must be non-negative, "
                f"got {pseudo_exterior_ring_weight}"
            )
        if pseudo_exterior_ring_radius < 0:
            raise ValueError(
                "pseudo_exterior_ring_radius must be non-negative, "
                f"got {pseudo_exterior_ring_radius}"
            )
        self.supervised_criterion = supervised_criterion
        self.pseudo_weight_ce = pseudo_weight_ce
        self.pseudo_weight_mask = pseudo_weight_mask
        self.pseudo_weight_dice = pseudo_weight_dice
        self.pseudo_no_object_weight = pseudo_no_object_weight
        self.pseudo_unmatched_negative_enabled = bool(pseudo_unmatched_negative_enabled)
        self.pseudo_unmatched_negative_weight = float(pseudo_unmatched_negative_weight)
        self.pseudo_unmatched_negative_score_thresh = float(pseudo_unmatched_negative_score_thresh)
        self.pseudo_exterior_ring_enabled = bool(pseudo_exterior_ring_enabled)
        self.pseudo_exterior_ring_weight = float(pseudo_exterior_ring_weight)
        self.pseudo_exterior_ring_radius = int(pseudo_exterior_ring_radius)

    def supervised_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute standard supervised loss via SetCriterion.

        Args:
            outputs: Model outputs with 'pred_logits' and 'pred_masks'
            targets: Ground truth targets

        Returns:
            Loss dict from SetCriterion (includes 'total_loss')
        """
        return self.supervised_criterion(outputs, targets)

    @torch.no_grad()
    def _match_pseudo_labels(
        self,
        s_logits: torch.Tensor,
        s_masks: torch.Tensor,
        t_labels: torch.Tensor,
        t_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hungarian-match student queries to teacher pseudo-labels.

        Args:
            s_logits: (Nq, C) student query classification logits.
            s_masks: (Nq, H, W) student query mask logits.
            t_labels: (N,) teacher pseudo-label class indices.
            t_masks: (N, H, W) teacher pseudo-label masks (sigmoid'd).

        Returns:
            src_indices: (N,) indices into student queries (matched).
            tgt_indices: (N,) indices into teacher pseudo-labels (matched).
        """
        device = s_logits.device

        # --- Classification cost: -log_softmax at teacher label positions ---
        # (Nq, N) — lower is better (student predicts teacher's class strongly)
        cost_cls = -F.log_softmax(s_logits.float(), dim=-1)[:, t_labels]

        # --- Mask cost: sigmoid BCE (pairwise, matmul trick for memory efficiency) ---
        # Same approach as HungarianMatcher._bce_cost to avoid O(Nq*N*H*W) expansion.
        s_flat = s_masks.float().flatten(1)  # (Nq, H*W)
        t_flat = t_masks.float().flatten(1)  # (N, H*W)
        hw = s_flat.shape[1]

        pos = F.binary_cross_entropy_with_logits(
            s_flat, torch.ones_like(s_flat), reduction="none"
        )  # (Nq, H*W)
        neg = F.binary_cross_entropy_with_logits(
            s_flat, torch.zeros_like(s_flat), reduction="none"
        )  # (Nq, H*W)
        cost_mask = (torch.mm(pos, t_flat.t()) + torch.mm(neg, (1 - t_flat).t())) / hw
        # cost_mask: (Nq, N)

        # --- Dice cost ---
        s_sig = s_masks.sigmoid().float().flatten(1)  # (Nq, H*W)
        numerator = 2 * torch.mm(s_sig, t_flat.t())  # (Nq, N)
        denominator = s_sig.sum(dim=1, keepdim=True) + t_flat.sum(dim=1, keepdim=True).t()
        cost_dice = 1 - (numerator + 1) / (denominator + 1)  # (Nq, N)

        # --- Combined cost ---
        C = cost_cls + cost_mask + cost_dice

        # Guard against NaN / Inf (e.g. empty masks)
        if not torch.isfinite(C).all():
            C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)

        # Hungarian matching on CPU (matrix is small: Nq x N)
        C_cpu = C.float().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(C_cpu)

        src_indices = torch.as_tensor(row_ind, dtype=torch.int64, device=device)
        tgt_indices = torch.as_tensor(col_ind, dtype=torch.int64, device=device)
        return src_indices, tgt_indices

    def _pseudo_label_loss_for_outputs(
        self,
        student_outputs: Dict[str, torch.Tensor],
        pseudo_targets: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        pred_logits = student_outputs["pred_logits"]  # (B, Nq, C)
        pred_masks = student_outputs["pred_masks"]  # (B, Nq, H, W)

        B = pred_logits.shape[0]
        device = pred_logits.device

        zero = pred_logits.reshape(-1)[:0].sum() + pred_masks.reshape(-1)[:0].sum()
        total_ce = zero
        total_mask = zero
        total_dice = zero
        total_matched = torch.tensor(0.0, dtype=pred_logits.dtype, device=device)
        total_unmatched_negative = zero
        total_unmatched_high_score = torch.tensor(0.0, dtype=pred_logits.dtype, device=device)
        total_exterior_ring = zero
        total_exterior_ring_matched = torch.tensor(0.0, dtype=pred_logits.dtype, device=device)
        total_exterior_ring_valid = torch.tensor(0.0, dtype=pred_logits.dtype, device=device)
        total_exterior_ring_zero = torch.tensor(0.0, dtype=pred_logits.dtype, device=device)
        total_exterior_ring_pixels = torch.tensor(0.0, dtype=pred_logits.dtype, device=device)
        total_exterior_ring_prob = zero

        for b in range(B):
            if b >= len(pseudo_targets):
                break

            pt = pseudo_targets[b]
            labels = pt.get("labels", None)
            masks = pt.get("masks", None)
            quality_scores = pt.get("quality_scores", None)

            if labels is None or masks is None or quality_scores is None:
                continue

            # Move teacher data to device once. Empty pseudo images are not
            # negative examples; they simply contribute no pseudo loss.
            t_labels = labels.to(device)
            t_masks = masks.to(device).float()
            q_all = quality_scores.to(device).float()

            if len(t_labels) == 0:
                continue

            t_masks_flat = t_masks.flatten(1)

            # Hungarian matching: find best student query for each teacher label.
            src_idx, tgt_idx = self._match_pseudo_labels(
                pred_logits[b],
                pred_masks[b],
                t_labels,
                t_masks,
            )
            if src_idx.numel() == 0:
                continue

            if self.pseudo_unmatched_negative_enabled:
                matched_mask = torch.zeros(pred_logits.shape[1], dtype=torch.bool, device=device)
                matched_mask[src_idx] = True
                scores = self._foreground_scores(pred_logits[b])
                selected = (~matched_mask) & (
                    scores >= self.pseudo_unmatched_negative_score_thresh
                )
                selected_count = int(selected.sum().item())
                if selected_count > 0:
                    background_class = int(pred_logits.shape[-1] - 1)
                    target_classes = torch.full(
                        (selected_count,),
                        background_class,
                        dtype=torch.long,
                        device=device,
                    )
                    negative_ce = F.cross_entropy(
                        pred_logits[b][selected].float(),
                        target_classes,
                        reduction="sum",
                    )
                    total_unmatched_negative = total_unmatched_negative + negative_ce.to(
                        dtype=pred_logits.dtype
                    )
                    total_unmatched_high_score = total_unmatched_high_score + torch.tensor(
                        float(selected_count),
                        dtype=pred_logits.dtype,
                        device=device,
                    )

            # Select matched student predictions and reorder teacher targets.
            s_logits = pred_logits[b][src_idx]  # (N, C)
            s_masks = pred_masks[b][src_idx]  # (N, H, W)
            s_masks_sig = s_masks.sigmoid().float().flatten(1)  # (N, H*W)
            t_masks_matched = t_masks_flat[tgt_idx]  # (N, H*W)
            t_labels_matched = t_labels[tgt_idx]  # (N,)

            # Quality scores reordered to match tgt_idx.
            q = q_all[tgt_idx].to(dtype=pred_logits.dtype)  # (N,)

            # 1. Classification loss: pseudo-labels are partial positives, so
            # only matched queries are supervised; unmatched queries are ignored.
            ce_loss_per_instance = F.cross_entropy(
                s_logits.float(), t_labels_matched, reduction="none"
            )  # (N,)
            total_ce = total_ce + (q * ce_loss_per_instance.to(q.dtype)).sum()

            # 2. Mask BCE loss. Teacher masks remain soft targets.
            mask_bce_per_instance = F.binary_cross_entropy_with_logits(
                s_masks.float().flatten(1), t_masks_matched, reduction="none"
            ).mean(
                dim=1
            )  # (N,)
            total_mask = total_mask + (q * mask_bce_per_instance.to(q.dtype)).sum()

            # 3. Dice loss. Teacher masks remain soft targets.
            numerator = 2 * (s_masks_sig * t_masks_matched).sum(-1)
            denominator = s_masks_sig.sum(-1) + t_masks_matched.sum(-1)
            dice_per_instance = 1 - (numerator + 1) / (denominator + 1)  # (N,)
            total_dice = total_dice + (q * dice_per_instance.to(q.dtype)).sum()

            if self.pseudo_exterior_ring_enabled:
                ring_masks, _, _ = self._exterior_ring_masks(
                    t_masks[tgt_idx].float(),
                    radius=self.pseudo_exterior_ring_radius,
                )
                matched_ring_count = torch.tensor(
                    float(src_idx.numel()),
                    dtype=pred_logits.dtype,
                    device=device,
                )
                total_exterior_ring_matched = (
                    total_exterior_ring_matched + matched_ring_count
                )
                for match_offset in range(int(src_idx.numel())):
                    ring_mask = ring_masks[match_offset]
                    ring_pixels = float(ring_mask.sum().item())
                    total_exterior_ring_pixels = total_exterior_ring_pixels + torch.tensor(
                        ring_pixels,
                        dtype=pred_logits.dtype,
                        device=device,
                    )
                    if ring_pixels <= 0.0:
                        total_exterior_ring_zero = total_exterior_ring_zero + torch.tensor(
                            1.0,
                            dtype=pred_logits.dtype,
                            device=device,
                        )
                        continue
                    ring_prob = s_masks[match_offset].sigmoid().float()[ring_mask].mean()
                    ring_prob = ring_prob.to(dtype=pred_logits.dtype)
                    total_exterior_ring = (
                        total_exterior_ring + q[match_offset].to(dtype=pred_logits.dtype) * ring_prob
                    )
                    total_exterior_ring_prob = total_exterior_ring_prob + ring_prob
                    total_exterior_ring_valid = total_exterior_ring_valid + torch.tensor(
                        1.0,
                        dtype=pred_logits.dtype,
                        device=device,
                    )

            total_matched = total_matched + torch.ones_like(q).sum()

        matched_norm = total_matched.clamp_min(1.0)

        losses = {
            "pseudo_loss_ce": self.pseudo_weight_ce * total_ce / matched_norm,
            "pseudo_loss_mask": self.pseudo_weight_mask * total_mask / matched_norm,
            "pseudo_loss_dice": self.pseudo_weight_dice * total_dice / matched_norm,
        }

        if self.pseudo_unmatched_negative_enabled:
            negative_norm = total_unmatched_high_score.clamp_min(1.0)
            losses["pseudo_unmatched_negative_loss"] = (
                self.pseudo_unmatched_negative_weight
                * total_unmatched_negative
                / negative_norm
            )
            losses["pseudo_unmatched_high_score_count"] = total_unmatched_high_score.detach()

        metric_keys = {"pseudo_unmatched_high_score_count"}

        if self.pseudo_exterior_ring_enabled:
            ring_norm = total_exterior_ring_valid.clamp_min(1.0)
            losses["pseudo_exterior_ring_loss"] = (
                self.pseudo_exterior_ring_weight * total_exterior_ring / ring_norm
            )
            losses["pseudo_exterior_ring_matched_count"] = (
                total_exterior_ring_matched.detach()
            )
            losses["pseudo_exterior_ring_valid_count"] = total_exterior_ring_valid.detach()
            losses["pseudo_exterior_ring_zero_ring_count"] = total_exterior_ring_zero.detach()
            losses["pseudo_exterior_ring_pixel_count"] = total_exterior_ring_pixels.detach()
            losses["pseudo_exterior_ring_prob"] = (
                total_exterior_ring_prob / ring_norm
            ).detach()
            metric_keys.update(
                {
                    "pseudo_exterior_ring_matched_count",
                    "pseudo_exterior_ring_valid_count",
                    "pseudo_exterior_ring_zero_ring_count",
                    "pseudo_exterior_ring_pixel_count",
                    "pseudo_exterior_ring_prob",
                }
            )

        loss_keys = [key for key in losses if key not in metric_keys]
        losses["pseudo_total"] = sum(losses[key] for key in loss_keys)
        return losses

    @staticmethod
    def _foreground_scores(logits: torch.Tensor) -> torch.Tensor:
        class_probs = F.softmax(logits.float(), dim=-1)
        if class_probs.shape[-1] > 1:
            return class_probs[:, :-1].max(dim=-1).values
        return class_probs[:, 0]

    @staticmethod
    def _distribution(values: List[float]) -> Dict[str, float | int]:
        if not values:
            return {"count": 0}
        tensor = torch.tensor(values, dtype=torch.float32)
        quantiles = torch.quantile(
            tensor,
            torch.tensor([0.25, 0.5, 0.75, 0.9, 0.95, 0.99], dtype=torch.float32),
        )
        return {
            "count": int(tensor.numel()),
            "min": float(tensor.min().item()),
            "mean": float(tensor.mean().item()),
            "p25": float(quantiles[0].item()),
            "median": float(quantiles[1].item()),
            "p50": float(quantiles[1].item()),
            "p75": float(quantiles[2].item()),
            "p90": float(quantiles[3].item()),
            "p95": float(quantiles[4].item()),
            "p99": float(quantiles[5].item()),
            "max": float(tensor.max().item()),
        }

    @staticmethod
    def _exterior_ring_masks(
        masks: torch.Tensor,
        *,
        radius: int,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if radius < 0:
            raise ValueError(f"exterior ring radius must be non-negative, got {radius}")
        if masks.ndim != 3:
            raise ValueError(f"masks must have shape (N, H, W), got {tuple(masks.shape)}")

        interior = masks.float() >= float(threshold)
        if radius == 0:
            dilated = interior
        else:
            dilated = (
                F.max_pool2d(
                    interior.float().unsqueeze(1),
                    kernel_size=2 * radius + 1,
                    stride=1,
                    padding=radius,
                ).squeeze(1)
                > 0
            )
        ring = dilated & (~interior)
        far = ~dilated
        return ring, interior, far

    @staticmethod
    def _threshold_key(threshold: float) -> str:
        return f"{float(threshold):.3f}"

    def _empty_exterior_ring_summary(self, radius: int) -> Dict[str, Any]:
        return {
            "radius": int(radius),
            "matched_count": 0,
            "gt_density_bucket_supported": False,
            "gt_density_bucket_note": "unsupported: target-unlabeled diagnostic batches do not expose GT count/image id",
            "exterior_ring_prob_distribution": self._distribution([]),
            "interior_prob_distribution": self._distribution([]),
            "background_far_prob_distribution": self._distribution([]),
            "pred_target_area_ratio_distribution": self._distribution([]),
            "ring_pixel_count_distribution": self._distribution([]),
            "exterior_ring_prob_values": [],
            "interior_prob_values": [],
            "background_far_prob_values": [],
            "pred_target_area_ratio_values": [],
            "ring_pixel_count_values": [],
            "per_density_bucket": {},
        }

    @torch.no_grad()
    def pseudo_label_diagnostics(
        self,
        student_outputs: Dict[str, torch.Tensor],
        pseudo_targets: List[Dict[str, Any]],
        *,
        high_score_thresholds: Tuple[float, ...] = (0.7, 0.9),
        exterior_ring_radius: int = 2,
    ) -> Dict[str, Any]:
        """Summarize pseudo-label matching and high-score unmatched queries.

        This method does not create training losses. It mirrors the main
        pseudo-label Hungarian matching path, then reports how many confident
        student queries are left unmatched by the pseudo-positive-only loss.
        """
        pred_logits = student_outputs["pred_logits"]
        pred_masks = student_outputs["pred_masks"]
        thresholds = tuple(float(value) for value in high_score_thresholds)
        for threshold in thresholds:
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError(f"high_score_threshold must be in [0, 1], got {threshold}")
        if exterior_ring_radius < 0:
            raise ValueError(
                f"exterior_ring_radius must be non-negative, got {exterior_ring_radius}"
            )

        image_count = min(int(pred_logits.shape[0]), len(pseudo_targets))
        query_count = int(pred_logits.shape[1]) if pred_logits.ndim >= 2 else 0

        unmatched_scores = {self._threshold_key(th): [] for th in thresholds}
        unmatched_max_ious = {self._threshold_key(th): [] for th in thresholds}
        high_without_pseudo_masks = {self._threshold_key(th): 0 for th in thresholds}
        ring_summary = self._empty_exterior_ring_summary(exterior_ring_radius)
        per_image = []

        total_kept = 0
        total_matched = 0
        total_unmatched = 0

        for b in range(image_count):
            pt = pseudo_targets[b]
            labels = pt["labels"]
            masks = pt["masks"]
            quality_scores = pt["quality_scores"]

            t_labels = labels.to(pred_logits.device)
            t_masks = masks.to(pred_masks.device).float()
            q_all = quality_scores.to(pred_logits.device).float()
            if t_labels.numel() != q_all.numel():
                raise ValueError(
                    "pseudo target labels and quality_scores must have matching lengths "
                    f"(got {t_labels.numel()} and {q_all.numel()})"
                )
            if t_masks.shape[0] != t_labels.numel():
                raise ValueError(
                    "pseudo target masks and labels must have matching lengths "
                    f"(got {t_masks.shape[0]} and {t_labels.numel()})"
                )

            kept_count = int(t_labels.numel())
            matched_idx = torch.empty(0, dtype=torch.int64, device=pred_logits.device)
            matched_tgt_idx = torch.empty(0, dtype=torch.int64, device=pred_logits.device)
            if kept_count > 0:
                matched_idx, matched_tgt_idx = self._match_pseudo_labels(
                    pred_logits[b],
                    pred_masks[b],
                    t_labels,
                    t_masks,
                )

            matched_mask = torch.zeros(query_count, dtype=torch.bool, device=pred_logits.device)
            if matched_idx.numel() > 0:
                matched_mask[matched_idx] = True
            unmatched_mask = ~matched_mask
            scores = self._foreground_scores(pred_logits[b])

            total_kept += kept_count
            total_matched += int(matched_idx.numel())
            total_unmatched += int(unmatched_mask.sum().item())

            if matched_idx.numel() > 0:
                matched_pred_prob = pred_masks[b][matched_idx].sigmoid().float()
                matched_target_masks = t_masks[matched_tgt_idx].float()
                ring_masks, interior_masks, far_masks = self._exterior_ring_masks(
                    matched_target_masks,
                    radius=int(exterior_ring_radius),
                )
                for match_offset in range(int(matched_idx.numel())):
                    pred_prob = matched_pred_prob[match_offset]
                    ring_mask = ring_masks[match_offset]
                    interior_mask = interior_masks[match_offset]
                    far_mask = far_masks[match_offset]
                    target_area = float(interior_mask.sum().item())
                    if target_area <= 0.0:
                        continue

                    ring_summary["matched_count"] += 1
                    ring_pixel_count = float(ring_mask.sum().item())
                    ring_summary["ring_pixel_count_values"].append(ring_pixel_count)
                    if ring_pixel_count > 0.0:
                        ring_summary["exterior_ring_prob_values"].append(
                            float(pred_prob[ring_mask].mean().item())
                        )
                    ring_summary["interior_prob_values"].append(
                        float(pred_prob[interior_mask].mean().item())
                    )
                    if bool(far_mask.any().item()):
                        ring_summary["background_far_prob_values"].append(
                            float(pred_prob[far_mask].mean().item())
                        )
                    pred_area = float((pred_prob >= 0.5).sum().item())
                    ring_summary["pred_target_area_ratio_values"].append(
                        pred_area / max(target_area, 1.0)
                    )

            image_record = {
                "image_index": b,
                "kept_pseudo_count": kept_count,
                "matched_query_count": int(matched_idx.numel()),
                "unmatched_query_count": int(unmatched_mask.sum().item()),
                "unmatched_high_score_counts": {},
            }

            max_iou_by_query = None
            if kept_count > 0:
                pred_flat = pred_masks[b].sigmoid().float().flatten(1)
                tgt_flat = t_masks.float().flatten(1)
                intersection = torch.mm(pred_flat, tgt_flat.t())
                union = (
                    pred_flat.sum(dim=1, keepdim=True)
                    + tgt_flat.sum(dim=1, keepdim=True).t()
                    - intersection
                )
                max_iou_by_query = (intersection / union.clamp_min(1e-6)).max(dim=1).values

            for threshold in thresholds:
                key = self._threshold_key(threshold)
                selected = unmatched_mask & (scores >= threshold)
                selected_count = int(selected.sum().item())
                image_record["unmatched_high_score_counts"][key] = selected_count
                if selected_count == 0:
                    continue
                unmatched_scores[key].extend(float(v) for v in scores[selected].detach().cpu().tolist())
                if max_iou_by_query is None:
                    high_without_pseudo_masks[key] += selected_count
                else:
                    unmatched_max_ious[key].extend(
                        float(v) for v in max_iou_by_query[selected].detach().cpu().tolist()
                    )

            per_image.append(image_record)

        for field in (
            "exterior_ring_prob",
            "interior_prob",
            "background_far_prob",
            "pred_target_area_ratio",
            "ring_pixel_count",
        ):
            ring_summary[f"{field}_distribution"] = self._distribution(
                ring_summary[f"{field}_values"]
            )

        return {
            "image_count": image_count,
            "query_count": query_count,
            "kept_pseudo_count": total_kept,
            "matched_query_count": total_matched,
            "unmatched_query_count": total_unmatched,
            "unmatched_high_score_counts": {
                key: len(values) for key, values in unmatched_scores.items()
            },
            "unmatched_high_score_score_distribution": {
                key: self._distribution(values) for key, values in unmatched_scores.items()
            },
            "unmatched_high_score_max_iou_distribution": {
                key: self._distribution(values) for key, values in unmatched_max_ious.items()
            },
            "unmatched_high_score_scores": unmatched_scores,
            "unmatched_high_score_max_ious": unmatched_max_ious,
            "unmatched_high_score_without_pseudo_mask_counts": high_without_pseudo_masks,
            "matched_exterior_ring": ring_summary,
            "per_image": per_image,
        }

    def pseudo_label_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        pseudo_targets: List[Dict[str, Any]],
        *,
        return_diagnostics: bool = False,
        high_score_thresholds: Tuple[float, ...] = (0.7, 0.9),
        exterior_ring_radius: int = 2,
    ) -> Dict[str, Any]:
        """
        Compute quality-weighted pseudo-label loss with Hungarian matching.

        Matched queries train against teacher pseudo-labels. Unmatched queries
        are ignored because pseudo-labels are partial positives. Auxiliary decoder
        outputs receive the same pseudo-label treatment with suffixed loss keys.

        Args:
            student_outputs: Student model outputs with 'pred_logits', 'pred_masks'
            pseudo_targets: List of dicts, each with:
                'labels': (N,) class labels from teacher
                'masks': (N, H, W) soft masks from teacher (sigmoid'd)
                'quality_scores': (N,) quality scores from PseudoLabelScorer

        Returns:
            Loss dict with pseudo classification, mask, dice, and total losses.
        """
        outputs_without_aux = {k: v for k, v in student_outputs.items() if k != "aux_outputs"}
        losses = self._pseudo_label_loss_for_outputs(outputs_without_aux, pseudo_targets)
        total = losses["pseudo_total"]

        for i, aux_outputs in enumerate(student_outputs.get("aux_outputs", [])):
            aux_losses = self._pseudo_label_loss_for_outputs(aux_outputs, pseudo_targets)
            for key, value in aux_losses.items():
                losses[f"{key}_{i}"] = value
            total = total + aux_losses["pseudo_total"]

        losses["pseudo_total"] = total
        if return_diagnostics:
            losses["pseudo_diagnostics"] = self.pseudo_label_diagnostics(
                outputs_without_aux,
                pseudo_targets,
                high_score_thresholds=high_score_thresholds,
                exterior_ring_radius=exterior_ring_radius,
            )
        return losses

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, Any]],
        pseudo_targets: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward: supervised + pseudo-label losses.

        Args:
            outputs: Student model outputs
            targets: Ground truth targets (labeled source)
            pseudo_targets: Teacher pseudo-labels (unlabeled target), optional

        Returns:
            Combined loss dict
        """
        # Supervised loss
        losses = self.supervised_loss(outputs, targets)

        # Pseudo-label loss (if available)
        if pseudo_targets is not None:
            pseudo_losses = self.pseudo_label_loss(outputs, pseudo_targets)
            for k, v in pseudo_losses.items():
                losses[k] = v

            # Add pseudo_total to total_loss
            if "pseudo_total" in losses and "total_loss" in losses:
                losses["total_loss"] = losses["total_loss"] + losses["pseudo_total"]

        return losses
