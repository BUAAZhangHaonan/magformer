from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    data = tensor.detach().float()
    if data.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": _round_float(data.min().item()),
        "max": _round_float(data.max().item()),
        "mean": _round_float(data.mean().item()),
        "std": _round_float(data.std(unbiased=False).item()),
    }


def compute_depth_sanity_report(
    *,
    depths: torch.Tensor,
    confidence_maps: Dict[str, torch.Tensor] | None = None,
    pred_masks: torch.Tensor | None = None,
) -> Dict[str, object]:
    report: Dict[str, object] = {
        "depth": _tensor_stats(depths),
        "confidence": {},
        "masks": {"foreground_ratio": None},
    }

    confidence_report: Dict[str, Dict[str, float]] = {}
    for key, value in (confidence_maps or {}).items():
        confidence_report[str(key)] = _tensor_stats(value)
    report["confidence"] = confidence_report

    if pred_masks is not None and pred_masks.numel() > 0:
        probs = pred_masks.detach().float()
        if probs.min().item() < 0.0 or probs.max().item() > 1.0:
            probs = probs.sigmoid()
        fg_ratio = (probs > 0.5).float().mean().item()
        report["masks"] = {"foreground_ratio": _round_float(fg_ratio)}

    return report


def should_abort_for_depth_sanity(
    report: Dict[str, object],
    *,
    min_depth_range: float = 0.05,
    min_confidence_range: float = 1e-5,
    min_mask_fg_ratio: float = 1e-3,
    max_mask_fg_ratio: float = 1.0 - 1e-3,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    depth_stats = dict(report.get("depth") or {})
    depth_min = float(depth_stats.get("min", 0.0))
    depth_max = float(depth_stats.get("max", 0.0))
    depth_range = depth_max - depth_min
    if depth_min < -1e-4 or depth_max > 1.0001:
        reasons.append(f"depth outside [0,1]: min={depth_min:.6f}, max={depth_max:.6f}")
    if depth_range < float(min_depth_range):
        reasons.append(
            f"depth range too narrow after normalization: range={depth_range:.6f}"
        )

    confidence = report.get("confidence") or {}
    if isinstance(confidence, dict):
        for key, stats in confidence.items():
            if not isinstance(stats, dict):
                continue
            conf_min = float(stats.get("min", 0.0))
            conf_max = float(stats.get("max", 0.0))
            conf_range = conf_max - conf_min
            if conf_range < float(min_confidence_range):
                reasons.append(
                    f"confidence map collapsed for {key}: range={conf_range:.6f}"
                )

    masks = dict(report.get("masks") or {})
    fg_ratio = masks.get("foreground_ratio")
    if fg_ratio is not None:
        fg_value = float(fg_ratio)
        if fg_value <= float(min_mask_fg_ratio):
            reasons.append(f"predicted masks are effectively empty: fg_ratio={fg_value:.6f}")
        if fg_value >= float(max_mask_fg_ratio):
            reasons.append(f"predicted masks are effectively full: fg_ratio={fg_value:.6f}")

    return bool(reasons), reasons


def write_depth_sanity_report(
    report: Dict[str, object],
    path: str | Path,
    *,
    reasons: Iterable[str] = (),
    aborted: bool = False,
) -> Path:
    out_path = Path(path).resolve()
    payload = dict(report)
    payload["aborted"] = bool(aborted)
    payload["reasons"] = list(reasons)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path
