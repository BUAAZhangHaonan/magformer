#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESOLUTIONS = [1024, 512]
MODEL_SPECS = [
    {"model_id": "magformer_depthnorm_on", "training_mode": "fine-tuned", "aliases": ["magformer_depthnorm_on"]},
    {"model_id": "magformer_lightdepth_convnextlite_spatialgate_edge_validhole", "training_mode": "fine-tuned", "aliases": ["magformer_lightdepth_convnextlite_spatialgate_edge_validhole"]},
    {"model_id": "magformer_lightdepth_mobilenetv3_sagate_edge_validhole", "training_mode": "fine-tuned", "aliases": ["magformer_lightdepth_mobilenetv3_sagate_edge_validhole"]},
    {"model_id": "magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole", "training_mode": "fine-tuned", "aliases": ["magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole"]},
    {
        "model_id": "magformer_nodpth_ref",
        "training_mode": "fine-tuned",
        "aliases": ["magformer_nodpth_ref_fair", "magformer_nodpth_ref"],
    },
    {"model_id": "mask2former", "training_mode": "fine-tuned", "aliases": ["mask2former", "official_mask2former_pretrained"]},
    {"model_id": "maskrcnn", "training_mode": "fine-tuned", "aliases": ["maskrcnn", "maskrcnn_pretrained"]},
    {"model_id": "mgm_mask2former_depthnorm_on", "training_mode": "fine-tuned", "aliases": ["mgm_mask2former_depthnorm_on"]},
    {"model_id": "mgm_mask2former_nodpth_ref", "training_mode": "fine-tuned", "aliases": ["mgm_mask2former_nodpth_ref"]},
    {"model_id": "msmformer", "training_mode": "from-scratch", "aliases": ["msmformer", "msmformer_scratch"]},
    {"model_id": "ucn", "training_mode": "from-scratch", "aliases": ["ucn", "ucn_scratch"]},
    {"model_id": "unet_boundary_inst", "training_mode": "from-scratch", "aliases": ["unet_boundary_inst"]},
    {"model_id": "unet_semantic_inst", "training_mode": "from-scratch", "aliases": ["unet_semantic_inst"]},
    {"model_id": "unetpp_boundary_inst", "training_mode": "from-scratch", "aliases": ["unetpp_boundary_inst"]},
    {"model_id": "uoais", "training_mode": "from-scratch", "aliases": ["uoais", "uoais_scratch"]},
    {"model_id": "yolov8_seg_l", "training_mode": "fine-tuned", "aliases": ["yolov8_seg_l_pretrained", "yolov8_seg_l"]},
    {"model_id": "yolov8_seg_m", "training_mode": "fine-tuned", "aliases": ["yolov8_seg_m_pretrained", "yolov8_seg_m"]},
    {"model_id": "yolov8_seg_n", "training_mode": "fine-tuned", "aliases": ["yolov8_seg_n_pretrained", "yolov8_seg_n"]},
    {"model_id": "yolov8_seg_s", "training_mode": "fine-tuned", "aliases": ["yolov8_seg_s_pretrained", "yolov8_seg_s"]},
    {"model_id": "yolov8_seg_x", "training_mode": "fine-tuned", "aliases": ["yolov8_seg_x_pretrained", "yolov8_seg_x"]},
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ffloat(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out):
        return None
    return out


def _read_metadata(model_dir: Path) -> Dict[str, Any]:
    path = model_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        payload = _load_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_inference_speed(model_dir: Path) -> Optional[Dict[str, Any]]:
    for name in ["inference_speed_clean.json", "inference_speed.json"]:
        path = model_dir / name
        if not path.exists():
            continue
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _read_wall_time(model_dir: Path, metadata: Dict[str, Any]) -> Optional[float]:
    path = model_dir / "wall_time_sec.txt"
    if path.exists():
        try:
            return float(path.read_text(encoding="utf-8").strip())
        except Exception:
            return None
    wall_time = _ffloat(metadata.get("wall_time_sec"))
    if wall_time is not None:
        return wall_time
    start = metadata.get("start_time_iso")
    end = metadata.get("end_time_iso")
    if isinstance(start, str) and isinstance(end, str):
        try:
            return float((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
        except Exception:
            return None
    return None


def _read_params(model_dir: Path) -> Optional[int]:
    path = model_dir / "params_trainable.txt"
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _read_peak_memory(model_dir: Path, inference: Optional[Dict[str, Any]]) -> Optional[float]:
    path = model_dir / "peak_memory_mb.txt"
    if path.exists():
        try:
            return float(path.read_text(encoding="utf-8").strip())
        except Exception:
            return None
    best: Optional[float] = None
    for cand in [model_dir / "log.txt", model_dir / "run.log"]:
        if not cand.exists():
            continue
        text = cand.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"max_mem:\s*(\d+)M", text):
            best = max(best or 0.0, float(match.group(1)))
        for match in re.finditer(r"\b\d+/\d+\s+([0-9]+(?:\.[0-9]+)?)G\b", text):
            best = max(best or 0.0, float(match.group(1)) * 1024.0)
    if best is not None:
        return best
    if isinstance(inference, dict):
        return _ffloat(inference.get("inference_peak_memory_mb"))
    return None


def _compute_prf50(annotation_path: Path, results_path: Path) -> Dict[str, Optional[float]]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception:
        return {"precision": None, "recall": None, "f1": None}

    if not annotation_path.exists() or not results_path.exists():
        return {"precision": None, "recall": None, "f1": None}

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(annotation_path))
    except Exception:
        return {"precision": None, "recall": None, "f1": None}

    try:
        rows = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return {"precision": None, "recall": None, "f1": None}
    if not isinstance(rows, list):
        return {"precision": None, "recall": None, "f1": None}
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(str(results_path))
            evaluator = COCOeval(coco_gt, coco_dt, "segm")
            evaluator.params.iouThrs = np.array([0.5], dtype=np.float64)
            evaluator.params.maxDets = [100]
            evaluator.evaluate()
            evaluator.accumulate()
    except Exception:
        return {"precision": None, "recall": None, "f1": None}

    precision = evaluator.eval.get("precision")
    if precision is None:
        return {"precision": None, "recall": None, "f1": None}
    precision_slice = precision[0, :, :, 0, 0]
    if precision_slice.ndim == 1:
        precision_slice = precision_slice[:, None]
    valid = precision_slice > -1
    if not np.any(valid):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    mean_precision = np.zeros(precision_slice.shape[0], dtype=np.float64)
    for idx in range(precision_slice.shape[0]):
        vals = precision_slice[idx][valid[idx]]
        mean_precision[idx] = float(vals.mean()) if vals.size else 0.0

    recall_thresholds = evaluator.params.recThrs
    denom = mean_precision + recall_thresholds
    f1_curve = np.divide(
        2.0 * mean_precision * recall_thresholds,
        denom,
        out=np.zeros_like(mean_precision, dtype=np.float64),
        where=denom > 0,
    )
    best_idx = int(np.argmax(f1_curve))
    return {
        "precision": float(mean_precision[best_idx] * 100.0),
        "recall": float(recall_thresholds[best_idx] * 100.0),
        "f1": float(f1_curve[best_idx] * 100.0),
    }


def _resolve_model_dirs(experiments_root: Path, resolution: int, aliases: Iterable[str]) -> List[Path]:
    matches: List[Path] = []
    for alias in aliases:
        matches.extend(
            path
            for path in experiments_root.glob(f"*{resolution}*full19/**/{alias}")
            if path.is_dir()
            and any(
                (path / marker).exists()
                for marker in (
                    "metrics.cocoeval.json",
                    "metadata.json",
                    "config.yaml",
                    "run.log",
                    "log.txt",
                )
            )
        )
    deduped = {path.resolve(): path.resolve() for path in matches}
    return sorted(deduped.values())


def _path_date_hint(path: Path) -> int:
    best = 0
    for match in re.finditer(r"(?<!\d)(20\d{6})(?!\d)", str(path)):
        best = max(best, int(match.group(1)))
    return best


def _metadata_command_mismatches_resolution(metadata: Dict[str, Any], resolution: int) -> bool:
    command = str(metadata.get("command", ""))
    if not command or "--image-size" not in command:
        return False
    return f"--image-size {resolution}" not in command


def _choose_model_dir(
    experiments_root: Path,
    resolution: int,
    candidates: List[Path],
    aliases: Iterable[str],
) -> Optional[Path]:
    if not candidates:
        return None
    alias_order = {alias: idx for idx, alias in enumerate(aliases)}

    def sort_key(path: Path) -> tuple[Any, ...]:
        rel = path.relative_to(experiments_root)
        rel_str = str(rel)
        metadata = _read_metadata(path)
        mismatch = _metadata_command_mismatches_resolution(metadata, resolution)
        alias_priority = alias_order.get(path.name, len(alias_order))
        date_hint = _path_date_hint(rel)
        return (
            1 if mismatch else 0,
            1 if "_backup" in rel_str else 0,
            1 if "_staging" in rel_str else 0,
            alias_priority,
            -date_hint,
            len(rel.parts),
            rel_str,
        )

    return sorted(candidates, key=sort_key)[0]


def _is_clean_alias_selection(
    experiments_root: Path,
    resolution: int,
    candidates: List[Path],
    chosen_model_dir: Optional[Path],
    aliases: Iterable[str],
) -> bool:
    if chosen_model_dir is None or len(candidates) <= 1:
        return False

    alias_order = {alias: idx for idx, alias in enumerate(aliases)}
    seen_alias_priorities: set[int] = set()
    chosen_priority: Optional[int] = None

    for path in candidates:
        rel = path.relative_to(experiments_root)
        rel_str = str(rel)
        metadata = _read_metadata(path)
        if _metadata_command_mismatches_resolution(metadata, resolution):
            return False
        if "_backup" in rel_str or "_staging" in rel_str:
            return False

        alias_priority = alias_order.get(path.name)
        if alias_priority is None:
            return False
        if alias_priority in seen_alias_priorities:
            return False
        seen_alias_priorities.add(alias_priority)

        if path == chosen_model_dir:
            chosen_priority = alias_priority

    return chosen_priority == min(seen_alias_priorities)


def _infer_training_mode(default_mode: str, metadata: Dict[str, Any], model_dir: Optional[Path]) -> str:
    command = str(metadata.get("command", ""))
    if "--pretrained" in command:
        return "fine-tuned"
    if "Warm-start loaded model weights" in _safe_read(model_dir / "run.log") if model_dir else False:
        return "fine-tuned"
    if "MODEL.WEIGHTS" in command or "model.finetune_weights" in command:
        return "fine-tuned"
    if "no pretrained weights" in _safe_read(model_dir / "run.log").lower() if model_dir else False:
        return "from-scratch"
    return default_mode


def _safe_read(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _build_note(
    *,
    status: str,
    multiple_candidates: List[Path],
    chosen_model_dir: Optional[Path],
    suppress_multiple_candidates_note: bool,
    metadata: Dict[str, Any],
    resolution: int,
    model_dir: Optional[Path],
    has_inference_metrics: bool,
    has_peak_memory: bool,
    has_train_wall_time: bool,
) -> str:
    notes: List[str] = []
    if status == "missing":
        notes.append("live artifact missing in current tree; rerun required")
    if len(multiple_candidates) > 1 and not suppress_multiple_candidates_note:
        notes.append(
            "multiple live artifact candidates found; using "
            + str(
                (chosen_model_dir or multiple_candidates[0]).relative_to(
                    model_dir.parents[2] if model_dir else (chosen_model_dir or multiple_candidates[0]).parents[2]
                )
            )
        )
    command = str(metadata.get("command", ""))
    if command and "--image-size" in command and f"--image-size {resolution}" not in command:
        notes.append(f"metadata command does not match resolution {resolution}")
    if status == "ok" and not has_train_wall_time:
        notes.append("no training-time artifact saved")
    if status == "ok" and not has_inference_metrics:
        notes.append("no inference-profile artifact saved")
    if status == "ok" and not has_peak_memory:
        notes.append("no training-memory artifact saved")
    return "; ".join(notes)


def _build_rows(repo_root: Path) -> List[Dict[str, Any]]:
    experiments_root = repo_root / "output" / "experiments"
    rows: List[Dict[str, Any]] = []

    for resolution in RESOLUTIONS:
        for spec in MODEL_SPECS:
            candidates = _resolve_model_dirs(experiments_root, resolution, spec["aliases"])
            model_dir = _choose_model_dir(experiments_root, resolution, candidates, spec["aliases"])
            metadata = _read_metadata(model_dir) if model_dir is not None else {}
            training_mode = _infer_training_mode(spec["training_mode"], metadata, model_dir)

            row: Dict[str, Any] = {
                "resolution": resolution,
                "model_id": spec["model_id"],
                "training_mode": training_mode,
                "status": "missing",
                "output_dir": str(model_dir) if model_dir is not None else "",
                "metrics_path": "",
                "results_path": "",
                "metadata_path": str(model_dir / "metadata.json") if model_dir is not None else "",
                "segm_AP": None,
                "segm_AP50": None,
                "segm_AP75": None,
                "bbox_AP": None,
                "bbox_AP50": None,
                "bbox_AP75": None,
                "segm_precision_at_50": None,
                "segm_recall_at_50": None,
                "segm_f1_at_50": None,
                "params_trainable": None,
                "train_wall_time_sec": None,
                "peak_memory_mb": None,
                "inference_latency_ms_mean": None,
                "inference_peak_memory_mb": None,
                "inference_fps": None,
                "note": "",
            }

            if model_dir is None:
                row["note"] = _build_note(
                    status="missing",
                    multiple_candidates=[],
                    chosen_model_dir=None,
                    suppress_multiple_candidates_note=False,
                    metadata={},
                    resolution=resolution,
                    model_dir=None,
                    has_inference_metrics=False,
                    has_peak_memory=False,
                    has_train_wall_time=False,
                )
                rows.append(row)
                continue

            metrics_path = model_dir / "metrics.cocoeval.json"
            results_path = model_dir / "coco_instances_results.json"
            inference = _read_inference_speed(model_dir)
            peak_memory = _read_peak_memory(model_dir, inference)
            train_wall_time = _read_wall_time(model_dir, metadata)
            row["metrics_path"] = str(metrics_path)
            row["results_path"] = str(results_path)
            row["params_trainable"] = _read_params(model_dir)
            row["train_wall_time_sec"] = train_wall_time
            row["peak_memory_mb"] = peak_memory
            row["inference_latency_ms_mean"] = _ffloat((inference or {}).get("latency_ms_mean"))
            row["inference_peak_memory_mb"] = _ffloat((inference or {}).get("inference_peak_memory_mb"))
            row["inference_fps"] = _ffloat((inference or {}).get("throughput_fps"))

            if metrics_path.exists():
                metrics = _load_json(metrics_path)
                row["segm_AP"] = _ffloat(metrics.get("segm/AP"))
                row["segm_AP50"] = _ffloat(metrics.get("segm/AP50"))
                row["segm_AP75"] = _ffloat(metrics.get("segm/AP75"))
                row["bbox_AP"] = _ffloat(metrics.get("bbox/AP"))
                row["bbox_AP50"] = _ffloat(metrics.get("bbox/AP50"))
                row["bbox_AP75"] = _ffloat(metrics.get("bbox/AP75"))
                dataset_root_raw = metadata.get("dataset_root")
                dataset_root = Path(dataset_root_raw) if isinstance(dataset_root_raw, str) and dataset_root_raw else None
                if dataset_root is not None:
                    prf = _compute_prf50(
                        dataset_root / "annotations" / "instances_val.json",
                        results_path,
                    )
                    row["segm_precision_at_50"] = prf["precision"]
                    row["segm_recall_at_50"] = prf["recall"]
                    row["segm_f1_at_50"] = prf["f1"]
                row["status"] = "ok"
            else:
                row["status"] = "partial"

            suppress_multiple_candidates_note = _is_clean_alias_selection(
                experiments_root,
                resolution,
                candidates,
                model_dir,
                spec["aliases"],
            )
            row["note"] = _build_note(
                status=row["status"],
                multiple_candidates=candidates,
                chosen_model_dir=model_dir,
                suppress_multiple_candidates_note=suppress_multiple_candidates_note,
                metadata=metadata,
                resolution=resolution,
                model_dir=model_dir,
                has_inference_metrics=inference is not None,
                has_peak_memory=peak_memory is not None,
                has_train_wall_time=train_wall_time is not None,
            )
            rows.append(row)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the live all-models two-resolutions metrics manifest.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    rows = _build_rows(repo_root)
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[live-metrics-manifest] wrote: {out_path}")


if __name__ == "__main__":
    main()
