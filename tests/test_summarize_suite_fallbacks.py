from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_min_metrics(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.cocoeval.json").write_text(
        json.dumps(
            {
                "iteration": 10,
                "segm/AP": 12.3,
                "segm/AP50": 30.0,
                "segm/AP75": 10.0,
                "bbox/AP": 11.1,
                "bbox/AP50": 25.0,
                "bbox/AP75": 9.0,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "coco_instances_results.json").write_text("[]", encoding="utf-8")


def test_summarize_suite_reads_inference_speed_clean_when_default_missing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "summarize_suite.py"
    out_root = tmp_path / "demo"
    model_dir = out_root / "maskrcnn"
    _write_min_metrics(model_dir)
    (model_dir / "inference_speed_clean.json").write_text(
        json.dumps({"status": "ok", "latency_ms_mean": 12.5, "throughput_fps": 80.0}),
        encoding="utf-8",
    )

    summary_name = "summary.json"
    subprocess.run(
        [sys.executable, str(script), "--output-root", str(out_root), "--write", "--write-name", summary_name],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((out_root / summary_name).read_text(encoding="utf-8"))
    inference = payload["maskrcnn"]["inference"]
    assert float(inference["latency_ms_mean"]) == 12.5
    assert float(inference["throughput_fps"]) == 80.0


def test_summarize_suite_falls_back_to_metadata_wall_time(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "summarize_suite.py"
    out_root = tmp_path / "demo"
    model_dir = out_root / "magformer_depthnorm_on"
    _write_min_metrics(model_dir)
    (model_dir / "metrics_log.jsonl").write_text(
        json.dumps({"phase": "val", "iter": 10, "val/segm_AP": 0.123, "val/bbox_AP": 0.111}) + "\n",
        encoding="utf-8",
    )
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "start_time_iso": "2026-03-24T00:00:00+08:00",
                "end_time_iso": "2026-03-24T01:30:00+08:00",
                "wall_time_sec": None,
            }
        ),
        encoding="utf-8",
    )

    summary_name = "summary.json"
    subprocess.run(
        [sys.executable, str(script), "--output-root", str(out_root), "--write", "--write-name", summary_name],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((out_root / summary_name).read_text(encoding="utf-8"))
    assert float(payload["magformer_depthnorm_on"]["wall_time_sec"]) == 5400.0


def test_summarize_suite_parses_peak_memory_from_logs_or_inference(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "summarize_suite.py"
    out_root = tmp_path / "demo"

    yolo_dir = out_root / "yolov8_seg_n"
    _write_min_metrics(yolo_dir)
    (yolo_dir / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "train" / "results.csv").write_text(
        "epoch,metrics/mAP50-95(M),metrics/mAP50(M),metrics/mAP50-95(B),metrics/mAP50(B)\n"
        "0,0.1,0.2,0.3,0.4\n",
        encoding="utf-8",
    )
    (yolo_dir / "run.log").write_text("Epoch    GPU_mem\n1/20      8.09G something\n", encoding="utf-8")

    unet_dir = out_root / "unet_boundary_inst"
    _write_min_metrics(unet_dir)
    (unet_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (unet_dir / "inference_speed.json").write_text(
        json.dumps({"status": "ok", "latency_ms_mean": 10.0, "throughput_fps": 100.0, "inference_peak_memory_mb": 512.0}),
        encoding="utf-8",
    )

    summary_name = "summary.json"
    subprocess.run(
        [sys.executable, str(script), "--output-root", str(out_root), "--write", "--write-name", summary_name],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((out_root / summary_name).read_text(encoding="utf-8"))
    assert float(payload["yolov8_seg_n"]["peak_memory_mb"]) > 8000.0
    assert float(payload["unet_boundary_inst"]["peak_memory_mb"]) == 512.0


def test_summarize_suite_uses_final_cocoeval_as_canonical_detectron2_publication_source(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "summarize_suite.py"
    out_root = tmp_path / "demo"
    model_dir = out_root / "mgm_mask2former_depthnorm_on"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.cocoeval.json").write_text(
        json.dumps(
            {
                "iteration": 999,
                "segm/AP": 72.8081,
                "segm/AP50": 87.9172,
                "segm/AP75": 78.6858,
                "bbox/AP": 61.5294,
                "bbox/AP50": 83.5271,
                "bbox/AP75": 69.8339,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "coco_instances_results.json").write_text("[]", encoding="utf-8")
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "iteration": 800,
                "segm/AP": 99.0,
                "segm/AP50": 99.0,
                "segm/AP75": 99.0,
                "bbox/AP": 0.0,
                "bbox/AP50": 0.0,
                "bbox/AP75": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary_name = "summary.json"
    subprocess.run(
        [sys.executable, str(script), "--output-root", str(out_root), "--write", "--write-name", summary_name],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((out_root / summary_name).read_text(encoding="utf-8"))
    entry = payload["mgm_mask2former_depthnorm_on"]
    assert float(entry["best"]["segm"]["AP"]) == 72.8081
    assert float(entry["best"]["bbox"]["AP"]) == 61.5294
    assert float(entry["last"]["bbox"]["AP"]) == 61.5294


def test_summarize_suite_keeps_exported_bbox_metrics_for_detectron2_rows(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "summarize_suite.py"
    out_root = tmp_path / "demo"

    model_dir = out_root / "mgm_mask2former_depthnorm_on"
    _write_min_metrics(model_dir)
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "iteration": 5,
                "segm/AP": 13.0,
                "segm/AP50": 31.0,
                "segm/AP75": 11.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (model_dir / "metrics.cocoeval.json").write_text(
        json.dumps(
            {
                "iteration": 10,
                "segm/AP": 72.81,
                "segm/AP50": 90.12,
                "segm/AP75": 79.33,
                "bbox/AP": 61.53,
                "bbox/AP50": 83.53,
                "bbox/AP75": 69.83,
            }
        ),
        encoding="utf-8",
    )

    summary_name = "summary.json"
    subprocess.run(
        [sys.executable, str(script), "--output-root", str(out_root), "--write", "--write-name", summary_name],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((out_root / summary_name).read_text(encoding="utf-8"))
    best_bbox = payload["mgm_mask2former_depthnorm_on"]["best"]["bbox"]
    assert float(best_bbox["AP"]) == 61.53
    assert float(best_bbox["AP50"]) == 83.53
    assert float(best_bbox["AP75"]) == 69.83
