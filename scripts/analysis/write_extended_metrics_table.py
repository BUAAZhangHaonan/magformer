#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_SUMMARIES = [
    "output/experiments/0831_1k_20ep_1024_depth_revisit/summary_0831_1k_20ep_1024_depth_revisit.json",
    "output/experiments/0831_1k_20ep_1024_lightdepth_stage_a/summary_0831_1k_20ep_1024_lightdepth_stage_a.json",
    "output/experiments/0831_1k_20ep_1024_lightdepth_stage_b/summary_0831_1k_20ep_1024_lightdepth_stage_b.json",
    "output/experiments/0831_1k_20ep_1024_lightdepth_stage_b_f5/summary_0831_1k_20ep_1024_lightdepth_stage_b_f5.json",
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(block: Dict[str, Any] | None, family: str, key: str) -> Any:
    if not isinstance(block, dict):
        return None
    metric_block = block.get(family)
    if not isinstance(metric_block, dict):
        return None
    return metric_block.get(key)


def _coalesce_entry(current: Dict[str, Any] | None, candidate: Dict[str, Any]) -> Dict[str, Any]:
    if current is None:
        return candidate
    score_current = sum(v is not None for v in current.values())
    score_candidate = sum(v is not None for v in candidate.values())
    if score_candidate > score_current:
        return candidate
    return current


def _build_rows(summary_paths: List[Path]) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for path in summary_paths:
        payload = _load_json(path)
        for model_id, entry in payload.items():
            if model_id in {"experiment", "output_root"} or not isinstance(entry, dict):
                continue
            if entry.get("status") not in {"ok", "partial"}:
                continue
            best = entry.get("best") or {}
            last = entry.get("last") or {}
            inference = entry.get("inference") or {}
            candidate = {
                "model_id": model_id,
                "status": entry.get("status"),
                "best_segm_AP": _metric(best, "segm", "AP"),
                "best_segm_AP50": _metric(best, "segm", "AP50"),
                "best_segm_AP75": _metric(best, "segm", "AP75"),
                "best_segm_APs": _metric(best, "segm", "APs"),
                "best_segm_APm": _metric(best, "segm", "APm"),
                "best_segm_APl": _metric(best, "segm", "APl"),
                "best_bbox_AP": _metric(best, "bbox", "AP"),
                "best_bbox_AP50": _metric(best, "bbox", "AP50"),
                "best_bbox_AP75": _metric(best, "bbox", "AP75"),
                "best_bbox_APs": _metric(best, "bbox", "APs"),
                "best_bbox_APm": _metric(best, "bbox", "APm"),
                "best_bbox_APl": _metric(best, "bbox", "APl"),
                "last_segm_AP": _metric(last, "segm", "AP"),
                "last_bbox_AP": _metric(last, "bbox", "AP"),
                "params_trainable": entry.get("params_trainable"),
                "train_wall_time_sec": entry.get("wall_time_sec"),
                "peak_memory_mb": entry.get("peak_memory_mb"),
                "inference_latency_ms_mean": inference.get("latency_ms_mean"),
                "inference_latency_ms_p50": inference.get("latency_ms_p50"),
                "inference_latency_ms_p90": inference.get("latency_ms_p90"),
                "inference_fps": inference.get("throughput_fps"),
                "inference_peak_memory_mb": inference.get("inference_peak_memory_mb"),
                "inference_status": inference.get("status"),
            }
            rows[model_id] = _coalesce_entry(rows.get(model_id), candidate)
    return sorted(
        rows.values(),
        key=lambda row: (
            row["best_segm_AP"] is None,
            -(row["best_segm_AP"] or -1.0),
            -(row["last_segm_AP"] or -1.0),
            -(row["inference_fps"] or -1.0),
            row["model_id"],
        ),
    )


def _to_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Model",
        "Best segm AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "Best bbox AP",
        "Last segm AP",
        "Params",
        "Train sec",
        "Peak mem MB",
        "Infer ms",
        "Infer FPS",
        "Infer status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["model_id"],
            row["best_segm_AP"],
            row["best_segm_AP50"],
            row["best_segm_AP75"],
            row["best_segm_APs"],
            row["best_segm_APm"],
            row["best_segm_APl"],
            row["best_bbox_AP"],
            row["last_segm_AP"],
            row["params_trainable"],
            row["train_wall_time_sec"],
            row["peak_memory_mb"],
            row["inference_latency_ms_mean"],
            row["inference_fps"],
            row["inference_status"],
        ]
        formatted = []
        for value in values:
            if isinstance(value, float):
                formatted.append(f"{value:.4f}")
            elif value is None:
                formatted.append("")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write extended metrics table from multiple summary jsons")
    parser.add_argument("--summary", action="append", default=[], help="Summary json path. Repeatable.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_paths = [Path(path).resolve() for path in (args.summary or DEFAULT_SUMMARIES)]
    rows = _build_rows(summary_paths)

    out_json = Path(args.out_json).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_md = Path(args.out_md).resolve()
    for path in [out_json, out_csv, out_md]:
        path.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    out_md.write_text(_to_markdown(rows), encoding="utf-8")
    print(f"[extended-metrics] wrote: {out_json}")
    print(f"[extended-metrics] wrote: {out_csv}")
    print(f"[extended-metrics] wrote: {out_md}")


if __name__ == "__main__":
    main()
