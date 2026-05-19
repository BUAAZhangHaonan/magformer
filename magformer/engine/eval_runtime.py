# -*- coding: utf-8 -*-
"""Shared inference-time evaluation helpers for trainer and CLI entrypoints."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
from torch.amp import autocast
import gc

from .coco_export import outputs_to_coco_instances
from .evaluator import COCOEvaluator
from .inference_stats import InferenceStatsAccumulator


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



def _slice_batch(batch: Dict[str, Any], limit: int) -> Dict[str, Any]:
    sliced = dict(batch)
    for key in ("images", "depths", "depth_valid_masks", "noise_masks", "padding_masks", "image_ids"):
        value = sliced.get(key)
        if value is None:
            continue
        if hasattr(value, "shape") and len(value.shape) > 0:
            sliced[key] = value[:limit]
        elif isinstance(value, (list, tuple)):
            sliced[key] = value[:limit]
    return sliced


def _loader_image_count(val_loader) -> int:
    dataset = getattr(val_loader, "dataset", None)
    if dataset is not None:
        return len(dataset)
    batch_size = int(getattr(val_loader, "batch_size", 1) or 1)
    return len(val_loader) * batch_size

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
    num_vis_images: int = 8,
    iou_types: Optional[List[str]] = None,
    max_images: Optional[int] = None,
    fail_on_empty: bool = False,
    dump_inference_stats: Optional[str | Path] = None,
    inference_topk: int = 100,
    max_dets: int = 100,
) -> EvaluationResult:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    is_primary = rank == 0
    inference_model = _get_inference_model(model)

    if iou_types is None:
        iou_types = ["bbox", "segm"]
    if max_images is not None and int(max_images) < 1:
        raise ValueError("max_images must be >= 1 when provided")
    if isinstance(inference_topk, bool) or int(inference_topk) <= 0:
        raise ValueError(f"inference_topk must be a positive integer, got {inference_topk!r}")
    if isinstance(max_dets, bool) or int(max_dets) <= 0:
        raise ValueError(f"max_dets must be a positive integer, got {max_dets!r}")
    inference_topk = int(inference_topk)
    max_dets = int(max_dets)
    total_images = min(int(max_images), _loader_image_count(val_loader)) if max_images is not None else _loader_image_count(val_loader)
    include_segmentation = "segm" in iou_types
    max_images_label = max_images if max_images is not None else "all"
    if is_primary:
        print(
            f"[Eval] Starting: iou_types={iou_types}, max_images={max_images_label} "
            f"({total_images} images), inference_topk={inference_topk}, max_dets={max_dets}"
        )

    evaluator = (
        COCOEvaluator(coco_gt=coco_gt, iou_types=iou_types, max_dets=max_dets)
        if coco_gt is not None
        else None
    )
    stats_accumulator = InferenceStatsAccumulator(coco_gt) if dump_inference_stats is not None else None

    local_scores: List[float] = []
    local_bbox_area_ratios: List[float] = []
    local_total_masks = 0
    local_nonempty_masks = 0
    local_total_preds = 0
    local_image_ids: List[int] = []
    num_vis_images = max(1, num_vis_images)
    vis_batches: List[Any] = []
    vis_outputs: List[Any] = []

    eval_start = time.time()
    num_evaluated = 0

    for batch in val_loader:
        if max_images is not None:
            remaining = int(max_images) - num_evaluated
            if remaining <= 0:
                break
            batch = _slice_batch(batch, remaining)
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        depth_valid_masks = batch.get("depth_valid_masks")
        if depth_valid_masks is not None:
            depth_valid_masks = depth_valid_masks.to(device)
        noise_masks = batch.get("noise_masks")
        if noise_masks is not None:
            noise_masks = noise_masks.to(device)
        padding_masks = batch.get("padding_masks")
        if padding_masks is not None:
            padding_masks = padding_masks.to(device)

        amp_context = autocast("cuda") if amp_enabled and device.type == "cuda" else nullcontext()
        forward_kwargs = {
            "padding_masks": padding_masks,
            "depth_noise_masks": noise_masks,
            "depth_valid_masks": depth_valid_masks,
        }
        if stats_accumulator is not None:
            forward_kwargs["collect_inference_stats"] = True
        with amp_context:
            outputs = inference_model.forward_inference_raw(
                images,
                depths,
                inference_topk=inference_topk,
                **forward_kwargs,
            )

        predictions = outputs_to_coco_instances(
            outputs=outputs,
            image_ids=batch.get("image_ids"),
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            category_offset=category_offset,
            category_ids=category_ids,
            include_segmentation=include_segmentation,
        )
        if stats_accumulator is not None:
            if "inference_stats" not in outputs:
                raise RuntimeError("dump_inference_stats requires model outputs to include 'inference_stats'")
            stats_accumulator.add_records(
                image_ids=batch.get("image_ids"),
                raw_stats=outputs["inference_stats"],
                predictions=outputs.get("predictions", []),
                exported_rows=predictions,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
            )
        if evaluator is not None:
            evaluator.update(predictions)
        if is_primary and len(vis_batches) < num_vis_images and len(predictions) > 0:
            vis_batches.append(batch)
            vis_outputs.append(outputs)
        local_total_preds += len(predictions)
        batch_image_ids = batch.get("image_ids")
        if batch_image_ids is None:
            batch_image_ids = []
        local_image_ids.extend(int(image_id) for image_id in batch_image_ids)

        for pred in predictions:
            score = pred.get("score")
            if score is not None:
                local_scores.append(float(score))
            mask = pred.get("mask")
            bbox = pred.get("bbox")
            if mask is None:
                continue
            local_total_masks += 1
            local_nonempty_masks += 1
            if bbox is not None and len(bbox) == 4:
                if isinstance(mask, dict) and "size" in mask:
                    h, w = int(mask["size"][0]), int(mask["size"][1])
                else:
                    h, w = int(max(1, float(bbox[3]) - float(bbox[1]))), int(max(1, float(bbox[2]) - float(bbox[0])))
                denom = float(max(1, h * w))
                x1, y1, x2, y2 = [float(v) for v in bbox]
                box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                local_bbox_area_ratios.append(box_area / denom)
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

        batch_image_count = int(images.shape[0])
        num_evaluated += batch_image_count
        if num_evaluated % 100 == 0 and is_primary:
            elapsed = time.time() - eval_start
            rate = num_evaluated / elapsed if elapsed > 0 else 0
            print(f"[Eval] Inference: {num_evaluated}/{total_images} "
                  f"({rate:.1f} img/s, elapsed {elapsed:.0f}s)")

        if max_images is not None and num_evaluated >= int(max_images):
            if is_primary:
                print(f"[Eval] Reached max_images={max_images}, stopping inference")
            break

    inference_time = time.time() - eval_start
    if is_primary:
        print(f"[Eval] Inference done: {num_evaluated} images in {inference_time:.1f}s "
              f"({num_evaluated/inference_time:.1f} img/s)")

    if evaluator is not None:
        evaluator.synchronize_between_processes()
    gathered_stats_records: List[Dict[str, Any]] = []
    if stats_accumulator is not None:
        gathered_stats_records = [
            dict(record)
            for record in _flatten_gathered_objects(_gather_object(stats_accumulator.records))
        ]

    total_preds = sum(int(v) for v in _gather_object(local_total_preds))
    eval_scores = [float(v) for v in _flatten_gathered_objects(_gather_object(local_scores))]
    eval_bbox_area_ratios = [
        float(v) for v in _flatten_gathered_objects(_gather_object(local_bbox_area_ratios))
    ]
    eval_total_masks = sum(int(v) for v in _gather_object(local_total_masks))
    eval_nonempty_masks = sum(int(v) for v in _gather_object(local_nonempty_masks))
    eval_image_ids = [int(v) for v in _flatten_gathered_objects(_gather_object(local_image_ids))]
    if evaluator is not None:
        evaluator.set_image_ids(eval_image_ids)

    if not is_primary:
        return EvaluationResult(
            log_dict={},
            coco_metrics={},
            coco_results_path=None,
            visualization_batch=None,
            visualization_outputs=None,
        )

    if dump_inference_stats is not None:
        stats_writer = InferenceStatsAccumulator(coco_gt)
        stats_writer.extend_records(gathered_stats_records)
        stats_path = stats_writer.dump(dump_inference_stats)
        print(f"[Eval] inference_stats_path={stats_path}")

    if fail_on_empty and evaluator is not None and total_preds == 0:
        raise RuntimeError(
            "Evaluation exported zero COCO predictions. Check checkpoint "
            "compatibility, model output keys, and score/mask thresholds."
        )

    log_dict: Dict[str, float] = {}
    coco_metrics: Dict[str, float] = {}
    coco_results_path: Optional[Path] = None

    if evaluator is not None:
        if is_primary:
            print(f"[Eval] Running pycocotools evaluation (iou_types={iou_types})...")
        eval_coco_start = time.time()
        coco_metrics = evaluator.summarize()
        eval_coco_time = time.time() - eval_coco_start
        if is_primary:
            print(f"[Eval] pycocotools done in {eval_coco_time:.1f}s")
        for key, value in coco_metrics.items():
            log_dict[f"val/{key}"] = float(value)
        if "segm_AP" in coco_metrics:
            log_dict["val/mAP"] = float(coco_metrics["segm_AP"])
        elif "bbox_AP" in coco_metrics:
            log_dict["val/mAP"] = float(coco_metrics["bbox_AP"])
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
        log_dict["val/diag_num_eval_images"] = float(len(set(eval_image_ids)))
        log_dict["val/diag_num_predictions"] = float(total_preds)
        coco_results_path = evaluator.dump(Path(output_dir) / "coco_instances_results.json")

    return EvaluationResult(
        log_dict=log_dict,
        coco_metrics=coco_metrics,
        coco_results_path=coco_results_path,
        visualization_batch=vis_batches[0] if vis_batches else None,
        visualization_outputs=vis_outputs[0] if vis_outputs else None,
    )
