from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_extended_metrics_table_manifest_mode_preserves_resolution_training_mode_and_notes(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "write_extended_metrics_table.py"

    manifest = [
        {
            "resolution": 1024,
            "model_id": "mask2former",
            "training_mode": "fine-tuned",
            "status": "ok",
            "segm_AP": 58.7554,
            "segm_AP50": 80.5301,
            "segm_AP75": 66.0488,
            "bbox_AP": 50.5990,
            "bbox_AP50": 77.8280,
            "bbox_AP75": 58.2970,
            "segm_precision_at_50": 93.4138,
            "segm_recall_at_50": 79.0,
            "segm_f1_at_50": 85.6044,
            "params_trainable": 44056196,
            "train_wall_time_sec": 6944.0,
            "peak_memory_mb": 35761.0,
            "inference_latency_ms_mean": 66.5025,
            "inference_peak_memory_mb": 2711.4526,
            "inference_fps": 15.0370,
            "note": "",
        },
        {
            "resolution": 512,
            "model_id": "mask2former",
            "training_mode": "fine-tuned",
            "status": "missing",
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
            "note": "live artifact missing in current tree; rerun required",
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    out_json = tmp_path / "extended.json"
    out_csv = tmp_path / "extended.csv"
    out_md = tmp_path / "extended.md"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = json.loads(out_json.read_text(encoding="utf-8"))
    assert rows[0]["resolution"] == 1024
    assert rows[0]["training_mode"] == "fine-tuned"
    assert rows[1]["resolution"] == 512
    assert rows[1]["note"] == "live artifact missing in current tree; rerun required"

    markdown = out_md.read_text(encoding="utf-8")
    assert "Resolution" in markdown
    assert "Training mode" in markdown
    assert "Infer mem MB" in markdown
    assert "live artifact missing in current tree; rerun required" in markdown
