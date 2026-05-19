#!/usr/bin/env python3
"""Dry-run VC-SUDA training-signal diagnostics without optimizer steps."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from magformer.config import load_config, set_seed  # noqa: E402
from tools import train as train_tool  # noqa: E402


class SignalDiagnosticsError(RuntimeError):
    """Raised when a required VC-SUDA diagnostic input is missing."""


def _resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _require_path(path: str | Path | None, label: str) -> Path:
    resolved = _resolve_project_path(path)
    if resolved is None:
        raise SignalDiagnosticsError(f"{label} is required")
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _require_tensor(batch: dict[str, Any], key: str) -> torch.Tensor:
    value = batch[key]
    if not torch.is_tensor(value):
        raise TypeError(f"batch[{key!r}] must be a tensor, got {type(value)}")
    return value


def _optional_tensor(batch: dict[str, Any], key: str, device: torch.device) -> torch.Tensor | None:
    value = batch.get(key)
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise TypeError(f"batch[{key!r}] must be a tensor when present, got {type(value)}")
    return value.to(device)


def _prepare_targets(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    prepared = []
    for target in targets:
        prepared.append(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
        )
    return prepared


def _tensor_to_float(value: Any) -> float | None:
    if not torch.is_tensor(value):
        return None
    if value.numel() != 1:
        return None
    return float(value.detach().float().cpu().item())


def summarize_loss_dict(losses: dict[str, Any]) -> dict[str, float]:
    summary = {}
    for key, value in losses.items():
        scalar = _tensor_to_float(value)
        if scalar is not None:
            summary[key] = scalar
    return summary


def _distribution(values: list[float]) -> dict[str, float | int]:
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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _build_pseudo_targets(filtered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pseudo_targets = []
    for result in filtered:
        pseudo_targets.append(
            {
                "labels": result["labels"],
                "masks": result["masks"],
                "quality_scores": result["scores"],
            }
        )
    return pseudo_targets


def _pseudo_count(result: dict[str, Any]) -> int:
    scores = result["scores"]
    if not torch.is_tensor(scores):
        raise TypeError(f"pseudo scores must be a tensor, got {type(scores)}")
    return int(scores.numel())


def _threshold_for_epoch(components: dict[str, Any], epoch: int) -> float:
    scheduler = components["curriculum_scheduler"]
    vc_cfg = components["vc_suda_config"]
    if scheduler is not None:
        return float(scheduler.get_threshold(epoch))
    return float(vc_cfg["pseudo_label"]["quality_threshold"])


def _unsupervised_weight_for_batch(
    components: dict[str, Any],
    *,
    current_batch: int,
    iters_per_epoch: int,
) -> float:
    vc_cfg = components["vc_suda_config"]
    scheduler = components["curriculum_scheduler"]
    max_weight = float(vc_cfg.get("unsupervised_weight", 1.0))
    warmup_iters = int(vc_cfg.get("unsupervised_warmup_iters", 0))
    if warmup_iters > 0:
        progress = min(1.0, float(current_batch + 1) / float(warmup_iters))
        return float(max_weight * (progress ** 2))
    if scheduler is None:
        return max_weight
    warmup_epochs = int(vc_cfg.get("unsupervised_warmup_epochs", 10))
    progress_epoch = float(current_batch + 1) / float(max(iters_per_epoch, 1))
    if progress_epoch >= warmup_epochs:
        return max_weight
    progress = progress_epoch / float(max(warmup_epochs, 1))
    return float(max_weight * (progress ** 2))


def _run_labeled_branch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    image_key: str,
    depth_key: str,
    annotation_key: str,
    padding_key: str,
    noise_key: str,
    validity_key: str,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    images = _require_tensor(batch, image_key).to(device)
    depths = _require_tensor(batch, depth_key).to(device)
    targets = _prepare_targets(batch[annotation_key], device)
    padding_masks = _optional_tensor(batch, padding_key, device)
    noise_masks = _optional_tensor(batch, noise_key, device)
    depth_valid_masks = _optional_tensor(batch, validity_key, device)

    amp_ctx = torch.autocast("cuda") if amp_enabled and device.type == "cuda" else nullcontext()
    with torch.no_grad(), amp_ctx:
        losses = model(
            images,
            depths,
            targets,
            padding_masks=padding_masks,
            depth_noise_masks=noise_masks,
            depth_valid_masks=depth_valid_masks,
        )
    if not isinstance(losses, dict) or "total_loss" not in losses:
        raise SignalDiagnosticsError(
            f"{image_key} forward did not return a loss dict with total_loss"
        )
    return losses


def _run_pseudo_branch(
    model: torch.nn.Module,
    components: dict[str, Any],
    batch: dict[str, Any],
    *,
    threshold: float,
    high_score_thresholds: tuple[float, ...],
    exterior_ring_radius: int,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    ema_teacher = components["ema_teacher"]
    scorer = components["pseudo_label_scorer"]
    criterion = components["criterion"]
    if ema_teacher is None or scorer is None:
        raise SignalDiagnosticsError("VC-SUDA Stage C diagnostics require EMA teacher and pseudo_label_scorer")

    target_weak_images = _require_tensor(batch, "target_weak_images").to(device)
    target_weak_depths = _require_tensor(batch, "target_weak_depths").to(device)
    target_weak_padding = _optional_tensor(batch, "target_weak_padding_masks", device)
    target_weak_noise = _optional_tensor(batch, "target_weak_noise_masks", device)
    target_weak_valid = _optional_tensor(batch, "target_weak_depth_valid_masks", device)

    with torch.no_grad():
        teacher_outputs = ema_teacher(
            target_weak_images,
            target_weak_depths,
            padding_masks=target_weak_padding,
            depth_noise_masks=target_weak_noise,
            depth_valid_masks=target_weak_valid,
        )
        scored = scorer.score(teacher_outputs, target_weak_depths)
        filtered = scorer.filter_by_threshold(scored, threshold)
        pseudo_targets = _build_pseudo_targets(filtered)

    target_strong_images = _require_tensor(batch, "target_strong_images").to(device)
    target_strong_depths = _require_tensor(batch, "target_strong_depths").to(device)
    target_strong_padding = _optional_tensor(batch, "target_strong_padding_masks", device)
    target_strong_noise = _optional_tensor(batch, "target_strong_noise_masks", device)
    target_strong_valid = _optional_tensor(batch, "target_strong_depth_valid_masks", device)

    amp_ctx = torch.autocast("cuda") if amp_enabled and device.type == "cuda" else nullcontext()
    with torch.no_grad(), amp_ctx:
        target_outputs = model(
            target_strong_images,
            target_strong_depths,
            targets=None,
            padding_masks=target_strong_padding,
            depth_noise_masks=target_strong_noise,
            depth_valid_masks=target_strong_valid,
            return_features=True,
        )
        pseudo_losses = criterion.pseudo_label_loss(
            target_outputs,
            pseudo_targets,
            return_diagnostics=True,
            high_score_thresholds=high_score_thresholds,
            exterior_ring_radius=exterior_ring_radius,
        )

    diagnostics = pseudo_losses.pop("pseudo_diagnostics")
    scored_count = sum(_pseudo_count(result) for result in scored)
    kept_count = sum(_pseudo_count(result) for result in filtered)
    empty_images = sum(1 for result in filtered if _pseudo_count(result) == 0)
    return {
        "losses": pseudo_losses,
        "diagnostics": diagnostics,
        "pseudo_scored_count": scored_count,
        "pseudo_kept_count": kept_count,
        "pseudo_empty_images": empty_images,
        "pseudo_keep_rate": float(kept_count) / float(scored_count) if scored_count else 0.0,
    }


def _aggregate(records: list[dict[str, Any]], thresholds: tuple[float, ...]) -> dict[str, Any]:
    branch_totals = {
        "source_total_loss": [],
        "target_labeled_total_loss_raw": [],
        "target_labeled_total_loss_weighted": [],
        "pseudo_total_raw": [],
        "pseudo_total_weighted": [],
    }
    pseudo_counts = {
        "pseudo_image_count": 0,
        "pseudo_scored_count": 0,
        "pseudo_instances_kept": 0,
        "matched_query_count": 0,
        "unmatched_query_count": 0,
    }
    high_counts = {f"{threshold:.3f}": 0 for threshold in thresholds}
    high_scores = {f"{threshold:.3f}": [] for threshold in thresholds}
    high_ious = {f"{threshold:.3f}": [] for threshold in thresholds}
    ring_radius = None
    ring_matched_count = 0
    ring_gt_density_bucket_supported = False
    ring_values = {
        "exterior_ring_prob": [],
        "interior_prob": [],
        "background_far_prob": [],
        "pred_target_area_ratio": [],
        "ring_pixel_count": [],
    }

    for record in records:
        branch_totals["source_total_loss"].append(record["source"]["total_loss"])
        branch_totals["target_labeled_total_loss_raw"].append(record["target_labeled"]["total_loss_raw"])
        branch_totals["target_labeled_total_loss_weighted"].append(record["target_labeled"]["total_loss_weighted"])
        branch_totals["pseudo_total_raw"].append(record["pseudo"]["total_loss_raw"])
        branch_totals["pseudo_total_weighted"].append(record["pseudo"]["total_loss_weighted"])

        diag = record["pseudo"]["diagnostics"]
        pseudo_counts["pseudo_image_count"] += int(diag["image_count"])
        pseudo_counts["pseudo_scored_count"] += int(record["pseudo"]["scored_count"])
        pseudo_counts["pseudo_instances_kept"] += int(diag["kept_pseudo_count"])
        pseudo_counts["matched_query_count"] += int(diag["matched_query_count"])
        pseudo_counts["unmatched_query_count"] += int(diag["unmatched_query_count"])
        for key in high_counts:
            high_counts[key] += int(diag["unmatched_high_score_counts"][key])
            high_scores[key].extend(diag["unmatched_high_score_scores"][key])
            high_ious[key].extend(diag["unmatched_high_score_max_ious"][key])
        ring = diag.get("matched_exterior_ring", {})
        if ring:
            ring_radius = int(ring.get("radius", ring_radius if ring_radius is not None else 2))
            ring_matched_count += int(ring.get("matched_count", 0))
            ring_gt_density_bucket_supported = bool(
                ring_gt_density_bucket_supported or ring.get("gt_density_bucket_supported", False)
            )
            for key in ring_values:
                ring_values[key].extend(float(v) for v in ring.get(f"{key}_values", []))

    return {
        "branch_loss_means": {key: _mean(values) for key, values in branch_totals.items()},
        "pseudo_counts": pseudo_counts,
        "unmatched_high_score_counts": high_counts,
        "unmatched_high_score_score_distribution": {
            key: _distribution(values) for key, values in high_scores.items()
        },
        "unmatched_high_score_max_iou_distribution": {
            key: _distribution(values) for key, values in high_ious.items()
        },
        "matched_exterior_ring": {
            "radius": int(ring_radius if ring_radius is not None else 2),
            "matched_count": int(ring_matched_count),
            "gt_density_bucket_supported": ring_gt_density_bucket_supported,
            "gt_density_bucket_note": (
                "unsupported: target-unlabeled diagnostic batches do not expose GT count/image id"
                if not ring_gt_density_bucket_supported
                else "supported"
            ),
            "exterior_ring_prob_distribution": _distribution(
                ring_values["exterior_ring_prob"]
            ),
            "interior_prob_distribution": _distribution(ring_values["interior_prob"]),
            "background_far_prob_distribution": _distribution(
                ring_values["background_far_prob"]
            ),
            "pred_target_area_ratio_distribution": _distribution(
                ring_values["pred_target_area_ratio"]
            ),
            "ring_pixel_count_distribution": _distribution(
                ring_values["ring_pixel_count"]
            ),
        },
    }


def run_diagnostics(
    config_path: str | Path,
    *,
    weights: str | Path | None,
    max_batches: int,
    batch_size: int | None,
    num_workers: int,
    device_name: str,
    high_score_thresholds: tuple[float, ...],
    exterior_ring_radius: int,
) -> dict[str, Any]:
    if max_batches <= 0:
        raise SignalDiagnosticsError("--max-batches must be positive")
    if batch_size is not None and batch_size <= 0:
        raise SignalDiagnosticsError("--batch-size must be positive when set")
    if exterior_ring_radius < 0:
        raise SignalDiagnosticsError("--exterior-ring-radius must be non-negative")

    os.chdir(PROJECT_ROOT)
    cfg_path = _require_path(config_path, "config")
    cfg = load_config(str(cfg_path))
    set_seed(int(cfg.runtime.seed))

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SignalDiagnosticsError(f"CUDA device requested but unavailable: {device_name}")

    effective_weights = _require_path(weights or getattr(cfg.model, "finetune_weights", None), "weights")
    train_dataset, val_dataset = train_tool.build_datasets(cfg)
    if getattr(train_dataset, "target_labeled", None) is None:
        raise SignalDiagnosticsError("VC-SUDA train dataset is missing target_labeled")
    if getattr(train_dataset, "target_unlabeled", None) is None:
        raise SignalDiagnosticsError("VC-SUDA train dataset is missing target_unlabeled")

    effective_batch_size = int(batch_size or cfg.solver.ims_per_batch)
    train_loader, _ = train_tool.build_data_loaders(
        cfg,
        train_dataset,
        val_dataset,
        batch_size=effective_batch_size,
        num_workers=num_workers,
        is_distributed=False,
    )

    model = train_tool.build_model(cfg, device)
    load_info = train_tool.load_finetune_weights(model, str(effective_weights), strict=False)
    components = train_tool._build_vc_suda_components(cfg, model, device)
    model.train()
    if components["ema_teacher"] is not None:
        components["ema_teacher"].eval()

    amp_enabled = bool(getattr(cfg.solver, "amp_enabled", False))
    iters_per_epoch = len(train_loader)
    records = []

    for batch_index, batch in enumerate(train_loader):
        if batch_index >= max_batches:
            break

        threshold = _threshold_for_epoch(components, epoch=0)
        unsup_weight = _unsupervised_weight_for_batch(
            components,
            current_batch=batch_index,
            iters_per_epoch=iters_per_epoch,
        )
        target_labeled_weight = float(components["vc_suda_config"].get("target_labeled_weight", 1.0))

        source_losses = _run_labeled_branch(
            model,
            batch,
            image_key="source_images",
            depth_key="source_depths",
            annotation_key="source_annotations",
            padding_key="source_padding_masks",
            noise_key="source_noise_masks",
            validity_key="source_depth_valid_masks",
            device=device,
            amp_enabled=amp_enabled,
        )
        target_labeled_losses = _run_labeled_branch(
            model,
            batch,
            image_key="target_labeled_images",
            depth_key="target_labeled_depths",
            annotation_key="target_labeled_annotations",
            padding_key="target_labeled_padding_masks",
            noise_key="target_labeled_noise_masks",
            validity_key="target_labeled_depth_valid_masks",
            device=device,
            amp_enabled=amp_enabled,
        )
        pseudo_result = _run_pseudo_branch(
            model,
            components,
            batch,
            threshold=threshold,
            high_score_thresholds=high_score_thresholds,
            exterior_ring_radius=exterior_ring_radius,
            device=device,
            amp_enabled=amp_enabled,
        )

        source_total = float(source_losses["total_loss"].detach().float().cpu().item())
        tl_total_raw = float(target_labeled_losses["total_loss"].detach().float().cpu().item())
        pseudo_total_raw = float(pseudo_result["losses"]["pseudo_total"].detach().float().cpu().item())
        records.append(
            {
                "batch_index": batch_index,
                "threshold": threshold,
                "unsupervised_weight": unsup_weight,
                "target_labeled_weight": target_labeled_weight,
                "source": {
                    "losses": summarize_loss_dict(source_losses),
                    "total_loss": source_total,
                },
                "target_labeled": {
                    "losses": summarize_loss_dict(target_labeled_losses),
                    "total_loss_raw": tl_total_raw,
                    "total_loss_weighted": tl_total_raw * target_labeled_weight,
                },
                "pseudo": {
                    "losses": summarize_loss_dict(pseudo_result["losses"]),
                    "total_loss_raw": pseudo_total_raw,
                    "total_loss_weighted": pseudo_total_raw * unsup_weight,
                    "scored_count": int(pseudo_result["pseudo_scored_count"]),
                    "kept_count": int(pseudo_result["pseudo_kept_count"]),
                    "empty_images": int(pseudo_result["pseudo_empty_images"]),
                    "keep_rate": float(pseudo_result["pseudo_keep_rate"]),
                    "diagnostics": pseudo_result["diagnostics"],
                },
            }
        )

    if not records:
        raise SignalDiagnosticsError("train loader produced zero diagnostic batches")

    return {
        "config": str(cfg_path),
        "weights": str(effective_weights),
        "device": str(device),
        "max_batches": int(max_batches),
        "batch_size": effective_batch_size,
        "num_workers": int(num_workers),
        "no_optimizer_step": True,
        "load_info": load_info,
        "high_score_thresholds": [float(value) for value in high_score_thresholds],
        "exterior_ring_radius": int(exterior_ring_radius),
        "aggregate": _aggregate(records, high_score_thresholds),
        "batches": records,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", "--finetune-weights", dest="weights", default=None)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--high-score-thresholds",
        type=float,
        nargs="+",
        default=[0.7, 0.9],
    )
    parser.add_argument("--exterior-ring-radius", type=int, default=2)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_diagnostics(
        args.config,
        weights=args.weights,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        high_score_thresholds=tuple(args.high_score_thresholds),
        exterior_ring_radius=args.exterior_ring_radius,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True)
    output_path = _resolve_project_path(args.output_json)
    if output_path is None:
        raise SignalDiagnosticsError("--output-json is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
    print(f"PASS vc_suda_signal_diagnostics output={output_path}")
    print(json.dumps(summary["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
