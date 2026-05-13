#!/usr/bin/env python3
"""Diagnose VC-SUDA Stage C pseudo-label keep-rate before training.

The gate loads a Stage C config, warm-starts the teacher from
``model.finetune_weights`` or ``--weights``, samples a small
``target_unlabeled`` batch, runs teacher eval forward, then reuses
``PseudoLabelScorer.score`` and ``filter_by_threshold`` from the training path.

Safe smoke usage:

    python tools/diagnose_vc_suda_pseudo_labels.py \
      --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
      --max-images 2 \
      --device cpu

It fails fast when no scored predictions are produced, or when the configured
threshold fails the keep-rate or empty-image gates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from magformer.config import load_config, set_seed  # noqa: E402
from magformer.models.common.pseudo_label_scorer import PseudoLabelScorer  # noqa: E402
from tools import train as train_tool  # noqa: E402

DEFAULT_THRESHOLD_SWEEP = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_MIN_KEEP_RATE = 0.10
DEFAULT_MAX_EMPTY_RATIO = 0.05


class PseudoLabelDiagnosticsError(RuntimeError):
    """Raised when the Stage C pseudo-label diagnostic gate fails."""

    def __init__(self, message: str, summary: dict[str, Any] | None = None):
        super().__init__(message)
        self.summary = summary


def _resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    return train_tool._cfg_get(obj, key, default)


def _tensor_count(result: dict[str, Any]) -> int:
    scores = result.get("scores")
    if scores is None:
        return 0
    return int(scores.numel())


def _scores_to_cpu(scored_results: list[dict[str, Any]]) -> torch.Tensor:
    tensors = []
    for result in scored_results:
        scores = result.get("scores")
        if torch.is_tensor(scores) and scores.numel() > 0:
            tensors.append(scores.detach().float().cpu())
    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(tensors)


def _score_distribution(scores: torch.Tensor) -> dict[str, float | int]:
    if scores.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(
        scores,
        torch.tensor([0.25, 0.5, 0.75, 0.9, 0.95, 0.99], dtype=torch.float32),
    )
    return {
        "count": int(scores.numel()),
        "min": float(scores.min().item()),
        "mean": float(scores.mean().item()),
        "p25": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "p75": float(quantiles[2].item()),
        "p90": float(quantiles[3].item()),
        "p95": float(quantiles[4].item()),
        "p99": float(quantiles[5].item()),
        "max": float(scores.max().item()),
    }


def summarize_threshold_sweep(
    scored_results: list[dict[str, Any]],
    *,
    thresholds: list[float] | tuple[float, ...],
) -> list[dict[str, Any]]:
    """Return keep-rate diagnostics for each threshold without applying the gate."""
    predictions = sum(_tensor_count(result) for result in scored_results)
    images = len(scored_results)
    sweep = []
    for threshold in thresholds:
        kept_per_image = []
        for result in scored_results:
            scores = result.get("scores")
            if torch.is_tensor(scores) and scores.numel() > 0:
                kept_per_image.append(int((scores >= float(threshold)).sum().item()))
            else:
                kept_per_image.append(0)

        kept = sum(kept_per_image)
        empty_images = sum(1 for count in kept_per_image if count == 0)
        sweep.append(
            {
                "threshold": float(threshold),
                "kept": kept,
                "keep_rate": kept / predictions if predictions else 0.0,
                "empty_images": empty_images,
                "empty_ratio": empty_images / images if images else 0.0,
                "kept_per_image_mean": kept / images if images else 0.0,
                "kept_per_image": kept_per_image,
                "kept_per_image_min": min(kept_per_image) if kept_per_image else 0,
                "kept_per_image_max": max(kept_per_image) if kept_per_image else 0,
                "zero_image_count": empty_images,
            }
        )
    return sweep


def summarize_pseudo_label_scores(
    scored_results: list[dict[str, Any]],
    filtered_results: list[dict[str, Any]],
    *,
    threshold: float,
    threshold_source: str,
    threshold_config: dict[str, Any] | None = None,
    min_keep_rate: float = DEFAULT_MIN_KEEP_RATE,
    max_empty_ratio: float = DEFAULT_MAX_EMPTY_RATIO,
) -> dict[str, Any]:
    """Return keep-rate diagnostics and fail on empty pseudo-label gates."""
    predictions = sum(_tensor_count(result) for result in scored_results)
    kept_per_image = [_tensor_count(result) for result in filtered_results]
    kept = sum(kept_per_image)
    keep_rate = kept / predictions if predictions else 0.0

    images = len(filtered_results)
    empty_images = sum(1 for count in kept_per_image if count == 0)
    empty_ratio = empty_images / images if images else 0.0
    summary = {
        "images": images,
        "predictions": predictions,
        "kept": kept,
        "keep_rate": keep_rate,
        "kept_per_image": kept_per_image,
        "empty_images": empty_images,
        "empty_ratio": empty_ratio,
        "score_distribution": _score_distribution(_scores_to_cpu(scored_results)),
        "threshold": {
            "value": float(threshold),
            "source": threshold_source,
            "config": threshold_config or {},
        },
        "gate": {
            "min_keep_rate": float(min_keep_rate),
            "max_empty_ratio": float(max_empty_ratio),
        },
    }
    if predictions == 0:
        raise PseudoLabelDiagnosticsError(
            "predictions=0 after PseudoLabelScorer.score; Stage C would train with no pseudo-label candidates.",
            summary=summary,
        )
    if keep_rate == 0.0:
        raise PseudoLabelDiagnosticsError(
            f"keep_rate=0 at threshold={threshold}; Stage C would keep zero pseudo-labels.",
            summary=summary,
        )
    if keep_rate < min_keep_rate:
        raise PseudoLabelDiagnosticsError(
            f"keep_rate={keep_rate:.6f} below min_keep_rate={min_keep_rate:.6f} at threshold={threshold}.",
            summary=summary,
        )
    if empty_ratio > max_empty_ratio:
        raise PseudoLabelDiagnosticsError(
            f"empty_ratio={empty_ratio:.6f} above max_empty_ratio={max_empty_ratio:.6f} at threshold={threshold}.",
            summary=summary,
        )
    return summary


def summarize_pseudo_label_diagnostics(
    scored_results: list[dict[str, Any]],
    filtered_results: list[dict[str, Any]],
    *,
    threshold: float,
    threshold_source: str,
    threshold_config: dict[str, Any] | None = None,
    threshold_sweep: list[float] | tuple[float, ...] = DEFAULT_THRESHOLD_SWEEP,
    min_keep_rate: float = DEFAULT_MIN_KEEP_RATE,
    max_empty_ratio: float = DEFAULT_MAX_EMPTY_RATIO,
) -> dict[str, Any]:
    """Return gate diagnostics and attach threshold sweep on pass or fail."""
    try:
        summary = summarize_pseudo_label_scores(
            scored_results,
            filtered_results,
            threshold=threshold,
            threshold_source=threshold_source,
            threshold_config=threshold_config,
            min_keep_rate=min_keep_rate,
            max_empty_ratio=max_empty_ratio,
        )
    except PseudoLabelDiagnosticsError as exc:
        if exc.summary is not None:
            exc.summary["threshold_sweep"] = summarize_threshold_sweep(
                scored_results,
                thresholds=threshold_sweep,
            )
        raise
    summary["threshold_sweep"] = summarize_threshold_sweep(scored_results, thresholds=threshold_sweep)
    return summary


def _resolve_threshold(
    cfg: Any,
    *,
    epoch: int,
    threshold_override: float | None,
) -> tuple[float, str, dict[str, Any]]:
    pl_cfg = cfg.vc_suda.pseudo_label
    threshold_config = {
        "quality_threshold": float(_cfg_get(pl_cfg, "quality_threshold", 0.5)),
        "use_curriculum": bool(_cfg_get(pl_cfg, "use_curriculum", True)),
        "epoch": int(epoch),
    }
    if threshold_override is not None:
        return (
            float(threshold_override),
            "--threshold",
            {**threshold_config, "override": float(threshold_override)},
        )

    if bool(_cfg_get(pl_cfg, "use_curriculum", True)):
        from magformer.models.common.curriculum import CurriculumScheduler

        cur_cfg = cfg.vc_suda.curriculum
        scheduler = CurriculumScheduler(
            start_threshold=float(_cfg_get(cur_cfg, "start_threshold", 0.7)),
            end_threshold=float(_cfg_get(cur_cfg, "end_threshold", 0.3)),
            warmup_epochs=int(_cfg_get(cur_cfg, "warmup_epochs", 15)),
        )
        value = float(scheduler.get_threshold(epoch))
        threshold_config.update(
            {
                "start_threshold": float(_cfg_get(cur_cfg, "start_threshold", 0.7)),
                "end_threshold": float(_cfg_get(cur_cfg, "end_threshold", 0.3)),
                "warmup_epochs": int(_cfg_get(cur_cfg, "warmup_epochs", 15)),
            }
        )
        return value, "vc_suda.curriculum", threshold_config

    return (
        float(_cfg_get(pl_cfg, "quality_threshold", 0.5)),
        "vc_suda.pseudo_label.quality_threshold",
        threshold_config,
    )


def _build_scorer(cfg: Any) -> PseudoLabelScorer:
    pl_cfg = cfg.vc_suda.pseudo_label
    return PseudoLabelScorer(max_instances=int(_cfg_get(pl_cfg, "max_instances", 100)))


def _slice_tensor(value: Any, count: int) -> Any:
    if torch.is_tensor(value) and value.ndim > 0:
        return value[:count]
    return value


def _target_weak_batch(batch: dict[str, Any], remaining: int) -> dict[str, Any]:
    images = batch.get("target_weak_images")
    depths = batch.get("target_weak_depths")
    if images is None or depths is None:
        raise PseudoLabelDiagnosticsError("batch is missing target_weak_images or target_weak_depths")
    count = min(int(images.shape[0]), remaining)
    return {
        "images": _slice_tensor(images, count),
        "depths": _slice_tensor(depths, count),
        "padding_masks": _slice_tensor(batch.get("target_weak_padding_masks"), count),
        "depth_noise_masks": _slice_tensor(batch.get("target_weak_noise_masks"), count),
    }


def _teacher_forward(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    images = batch["images"].to(device)
    depths = batch["depths"].to(device)
    padding_masks = batch["padding_masks"]
    depth_noise_masks = batch["depth_noise_masks"]
    if torch.is_tensor(padding_masks):
        padding_masks = padding_masks.to(device)
    if torch.is_tensor(depth_noise_masks):
        depth_noise_masks = depth_noise_masks.to(device)

    if hasattr(model, "forward_inference_decoder_outputs"):
        return model.forward_inference_decoder_outputs(
            images,
            depths,
            padding_masks=padding_masks,
            depth_noise_masks=depth_noise_masks,
        )
    return model(
        images,
        depths,
        targets=None,
        padding_masks=padding_masks,
        depth_noise_masks=depth_noise_masks,
        return_features=True,
    )


def run_diagnostics(
    config_path: str | Path,
    *,
    weights: str | Path | None = None,
    max_images: int = 2,
    batch_size: int = 1,
    num_workers: int = 0,
    device_name: str = "cpu",
    epoch: int = 0,
    threshold_override: float | None = None,
    threshold_sweep: list[float] | tuple[float, ...] = DEFAULT_THRESHOLD_SWEEP,
    min_keep_rate: float = DEFAULT_MIN_KEEP_RATE,
    max_empty_ratio: float = DEFAULT_MAX_EMPTY_RATIO,
) -> dict[str, Any]:
    if max_images <= 0:
        raise PseudoLabelDiagnosticsError("--max-images must be positive")
    if batch_size <= 0:
        raise PseudoLabelDiagnosticsError("--batch-size must be positive")

    os.chdir(PROJECT_ROOT)
    cfg_path = _resolve_project_path(config_path)
    if cfg_path is None:
        raise PseudoLabelDiagnosticsError("config path is required")
    cfg = load_config(str(cfg_path))
    set_seed(int(cfg.runtime.seed))

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PseudoLabelDiagnosticsError(f"CUDA device requested but unavailable: {device_name}")

    effective_weights = _resolve_project_path(weights or getattr(cfg.model, "finetune_weights", None))
    if effective_weights is None:
        raise PseudoLabelDiagnosticsError("weights are required via --weights or model.finetune_weights")
    if not effective_weights.exists():
        raise PseudoLabelDiagnosticsError(f"weights not found: {effective_weights}")

    train_dataset, val_dataset = train_tool.build_datasets(cfg)
    if getattr(train_dataset, "target_unlabeled", None) is None:
        raise PseudoLabelDiagnosticsError("Stage C dataset did not build target_unlabeled")
    train_loader, _ = train_tool.build_data_loaders(
        cfg,
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        is_distributed=False,
    )

    model = train_tool.build_model(cfg, device)
    train_tool.load_finetune_weights(model, str(effective_weights), strict=False)
    model.eval()

    scorer = _build_scorer(cfg)
    threshold, threshold_source, threshold_config = _resolve_threshold(
        cfg,
        epoch=epoch,
        threshold_override=threshold_override,
    )

    scored_all: list[dict[str, Any]] = []
    filtered_all: list[dict[str, Any]] = []
    seen = 0
    with torch.inference_mode():
        for raw_batch in train_loader:
            remaining = max_images - seen
            if remaining <= 0:
                break
            batch = _target_weak_batch(raw_batch, remaining)
            teacher_outputs = _teacher_forward(model, batch, device=device)
            depths = batch["depths"].to(device)
            scored = scorer.score(teacher_outputs, depths)
            filtered = scorer.filter_by_threshold(scored, threshold)
            scored_all.extend(scored)
            filtered_all.extend(filtered)
            seen += int(batch["images"].shape[0])

    if seen == 0:
        raise PseudoLabelDiagnosticsError("target_unlabeled loader produced zero images")

    summary = summarize_pseudo_label_diagnostics(
        scored_all,
        filtered_all,
        threshold=threshold,
        threshold_source=threshold_source,
        threshold_config=threshold_config,
        threshold_sweep=threshold_sweep,
        min_keep_rate=min_keep_rate,
        max_empty_ratio=max_empty_ratio,
    )
    summary.update(
        {
            "config": str(cfg_path),
            "weights": str(effective_weights),
            "device": str(device),
            "max_images": int(max_images),
            "batch_size": int(batch_size),
            "target_unlabeled_ann": str(cfg.vc_suda.target_unlabeled_ann),
        }
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vc_suda_stage_c_1024_teacher8499.yaml")
    parser.add_argument("--weights", "--finetune-weights", dest="weights", default=None)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch used only when curriculum is enabled.")
    parser.add_argument("--threshold", type=float, default=None, help="Override the config threshold for probing.")
    parser.add_argument(
        "--min-keep-rate",
        type=float,
        default=DEFAULT_MIN_KEEP_RATE,
        help="Fail when the effective threshold keeps less than this fraction of scored predictions.",
    )
    parser.add_argument(
        "--max-empty-ratio",
        type=float,
        default=DEFAULT_MAX_EMPTY_RATIO,
        help="Fail when more than this fraction of sampled images keep zero pseudo-labels.",
    )
    parser.add_argument(
        "--threshold-sweep",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLD_SWEEP),
        help="Thresholds to scan and report without changing the fail-fast gate.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write the diagnostic JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = run_diagnostics(
            args.config,
            weights=args.weights,
            max_images=args.max_images,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
            epoch=args.epoch,
            threshold_override=args.threshold,
            threshold_sweep=args.threshold_sweep,
            min_keep_rate=args.min_keep_rate,
            max_empty_ratio=args.max_empty_ratio,
        )
    except PseudoLabelDiagnosticsError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        if exc.summary is not None:
            payload = json.dumps(exc.summary, sort_keys=True)
            print(payload)
            if args.output_json:
                output_path = _resolve_project_path(args.output_json)
                assert output_path is not None
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(payload + "\n", encoding="utf-8")
        return 1

    payload = json.dumps(summary, sort_keys=True)
    print("PASS pseudo_label_diagnostics")
    print(payload)
    if args.output_json:
        output_path = _resolve_project_path(args.output_json)
        assert output_path is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
