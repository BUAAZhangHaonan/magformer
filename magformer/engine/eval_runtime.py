# -*- coding: utf-8 -*-
"""Shared inference-time evaluation helpers for trainer and CLI entrypoints."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
from torch.amp import autocast

from .coco_export import outputs_to_coco_instances
from .evaluator import COCOEvaluator


@dataclass
class EvaluationResult:
    log_dict: Dict[str, float]
    coco_metrics: Dict[str, float]
    coco_results_path: Optional[Path]
    visualization_batch: Optional[Dict[str, Any]]
    visualization_outputs: Optional[Dict[str, Any]]


def _flatten_gathered_objects(gathered: Iterable[Any]) -> List[Any]:
    flattened: List[Any] = []
    for item in gathered:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _gather_object(value: Any) -> List[Any]:
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def _get_inference_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def run_inference_evaluation(
    model: torch.nn.Module,
    val_loader,
    *,
    coco_gt: Optional[Any],
    device: torch.device,
    output_dir: str | Path,
    amp_enabled: bool,
    score_threshold: float = 0.05,
    mask_threshold: float = 0.5,
    category_offset: int = 1,
    category_ids: Optional[List[int]] = None,
) -> EvaluationResult:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    is_primary = rank == 0
    inference_model = _get_inference_model(model)
    evaluator = (
        COCOEvaluator(coco_gt=coco_gt, iou_types=["bbox", "segm"], max_dets=100)
        if coco_gt is not None
        else None
    )

    local_scores: List[float] = []
    local_bbox_area_ratios: List[float] = []
    local_total_masks = 0
    local_nonempty_masks = 0
    local_total_preds = 0
    visualization_batch = None
    visualization_outputs = None

    for batch in val_loader:
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        noise_masks = batch.get("noise_masks")
        if noise_masks is not None:
            noise_masks = noise_masks.to(device)
        padding_masks = batch.get("padding_masks")
        if padding_masks is not None:
            padding_masks = padding_masks.to(device)

        amp_context = autocast("cuda") if amp_enabled and device.type == "cuda" else nullcontext()
        with amp_context:
            outputs = inference_model.forward_inference_raw(
                images,
                depths,
                padding_masks=padding_masks,
                depth_noise_masks=noise_masks,
            )

        if is_primary and visualization_outputs is None:
            visualization_batch = batch
            visualization_outputs = outputs

        predictions = outputs_to_coco_instances(
            outputs=outputs,
            image_ids=batch.get("image_ids"),
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            category_offset=category_offset,
            category_ids=category_ids,
        )
        if evaluator is not None:
            evaluator.update(predictions)
        local_total_preds += len(predictions)

        for pred in predictions:
            score = pred.get("score")
            if score is not None:
                local_scores.append(float(score))
            mask = pred.get("mask")
            bbox = pred.get("bbox")
            if mask is None:
                continue
            mask_arr = mask
            if hasattr(mask_arr, "detach") and hasattr(mask_arr, "cpu"):
                mask_arr = mask_arr.detach().cpu().numpy()
            elif hasattr(mask_arr, "cpu") and hasattr(mask_arr, "numpy"):
                mask_arr = mask_arr.cpu().numpy()
            local_total_masks += 1
            if float(mask_arr.sum()) > 0.0:
                local_nonempty_masks += 1
            if bbox is not None and len(bbox) == 4 and getattr(mask_arr, "ndim", 0) >= 2:
                h, w = int(mask_arr.shape[-2]), int(mask_arr.shape[-1])
                denom = float(max(1, h * w))
                x1, y1, x2, y2 = [float(v) for v in bbox]
                box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                local_bbox_area_ratios.append(box_area / denom)

    if evaluator is not None:
        evaluator.synchronize_between_processes()

    total_preds = sum(int(v) for v in _gather_object(local_total_preds))
    eval_scores = [float(v) for v in _flatten_gathered_objects(_gather_object(local_scores))]
    eval_bbox_area_ratios = [
        float(v) for v in _flatten_gathered_objects(_gather_object(local_bbox_area_ratios))
    ]
    eval_total_masks = sum(int(v) for v in _gather_object(local_total_masks))
    eval_nonempty_masks = sum(int(v) for v in _gather_object(local_nonempty_masks))

    if not is_primary:
        return EvaluationResult(
            log_dict={},
            coco_metrics={},
            coco_results_path=None,
            visualization_batch=None,
            visualization_outputs=None,
        )

    log_dict: Dict[str, float] = {}
    coco_metrics: Dict[str, float] = {}
    coco_results_path: Optional[Path] = None

    if evaluator is not None:
        coco_metrics = evaluator.summarize()
        for key, value in coco_metrics.items():
            log_dict[f"val/{key}"] = float(value)
        if "segm_AP" in coco_metrics:
            log_dict["val/mAP"] = float(coco_metrics["segm_AP"])
        if eval_scores:
            import numpy as np

            score_arr = np.asarray(eval_scores, dtype=np.float64)
            log_dict["val/diag_score_p50"] = float(np.percentile(score_arr, 50))
            log_dict["val/diag_score_p90"] = float(np.percentile(score_arr, 90))
        if eval_bbox_area_ratios:
            import numpy as np

            bbox_arr = np.asarray(eval_bbox_area_ratios, dtype=np.float64)
            log_dict["val/diag_bbox_area_ratio_p50"] = float(np.percentile(bbox_arr, 50))
            log_dict["val/diag_bbox_area_ratio_p90"] = float(np.percentile(bbox_arr, 90))
        if eval_total_masks > 0:
            log_dict["val/diag_mask_nonempty_ratio"] = float(
                eval_nonempty_masks / float(eval_total_masks)
            )
        log_dict["val/diag_num_predictions"] = float(total_preds)
        coco_results_path = evaluator.dump(Path(output_dir) / "coco_instances_results.json")

    return EvaluationResult(
        log_dict=log_dict,
        coco_metrics=coco_metrics,
        coco_results_path=coco_results_path,
        visualization_batch=visualization_batch,
        visualization_outputs=visualization_outputs,
    )
