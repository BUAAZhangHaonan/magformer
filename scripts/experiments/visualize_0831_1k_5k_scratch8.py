#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--output-root", type=str, required=True)
    ap.add_argument("--ann-file", type=str, default="annotations/instances_val.json")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--num-images", type=int, default=50)
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--show-labels", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root)
    summary_path = output_root / "summary_0831_1k_5k_scratch8.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary json not found: {summary_path}")

    summary: Dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))

    vis_root = output_root / "visualizations"
    vis_root.mkdir(parents=True, exist_ok=True)

    # 1) Per-model overlay visualizations (8 models)
    for model_id, model_info in summary.items():
        if not model_id.endswith("_scratch"):
            continue
        if not isinstance(model_info, dict) or model_info.get("status") != "ok":
            print(f"[vis] skip {model_id}: status={getattr(model_info, 'get', lambda *_: None)('status')}")
            continue

        artifacts = model_info.get("artifacts", {})
        results_json = artifacts.get("coco_instances_results")
        if not results_json:
            print(f"[vis] skip {model_id}: missing artifacts.coco_instances_results")
            continue

        results_json_path = Path(results_json)
        if not results_json_path.exists():
            # Backward compatibility for historical detectron2 outputs.
            legacy = output_root / model_id / "inference" / "coco_instances_results.json"
            if legacy.exists():
                results_json_path = legacy
        if not results_json_path.exists():
            print(f"[vis] skip {model_id}: results json not found: {results_json_path}")
            continue

        out_dir = vis_root / model_id / "overlay"
        out_dir.mkdir(parents=True, exist_ok=True)

        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "visualization" / "visualize_coco_results.py"),
                "--dataset-root",
                args.dataset_root,
                "--ann-file",
                args.ann_file,
                "--split",
                args.split,
                "--results-json",
                str(results_json_path),
                "--output-dir",
                str(out_dir),
                "--num-images",
                str(args.num_images),
                "--score-threshold",
                str(args.score_threshold),
                "--alpha",
                str(args.alpha),
                *(["--show-labels"] if args.show_labels else []),
                "--prefix",
                "overlay",
            ]
        )

    # 2) Mandatory triptych: GT / MagFormer / MGM_Mask2Former
    mag = summary.get("magformer_scratch", {})
    mgm = summary.get("mgm_mask2former_scratch", {})
    mag_json = (mag.get("artifacts", {}) or {}).get("coco_instances_results")
    mgm_json = (mgm.get("artifacts", {}) or {}).get("coco_instances_results")
    if mag_json and not Path(mag_json).exists():
        fallback = output_root / "magformer_scratch" / "inference" / "coco_instances_results.json"
        mag_json = str(fallback) if fallback.exists() else mag_json
    if mgm_json and not Path(mgm_json).exists():
        fallback = output_root / "mgm_mask2former_scratch" / "inference" / "coco_instances_results.json"
        mgm_json = str(fallback) if fallback.exists() else mgm_json
    if mag_json and mgm_json and Path(mag_json).exists() and Path(mgm_json).exists():
        trip_out = vis_root / "triptych_gt_magformer_mgm"
        trip_out.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "compare_predictions.py"),
                "--dataset-root",
                args.dataset_root,
                "--ann-file",
                args.ann_file,
                "--split",
                args.split,
                "--magformer-results",
                str(Path(mag_json)),
                "--mask2former-results",
                str(Path(mgm_json)),
                "--output-dir",
                str(trip_out),
                "--num-images",
                str(args.num_images),
                "--score-threshold",
                str(args.score_threshold),
                "--alpha",
                str(args.alpha),
                *(["--show-labels"] if args.show_labels else []),
            ]
        )
    else:
        print("[vis] skip triptych: magformer_scratch or mgm_mask2former_scratch results missing")

    print(f"[vis] done. outputs under: {vis_root}")


if __name__ == "__main__":
    # Ensure local imports from repo root work when invoked from anywhere.
    os.chdir(Path(__file__).resolve().parents[2])
    main()
