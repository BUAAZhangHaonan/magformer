from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_extended_metrics_table_sorts_by_best_then_last_then_fps(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "write_extended_metrics_table.py"

    summary = {
        "experiment": "demo",
        "output_root": str(tmp_path / "out"),
        "model_a": {
            "status": "ok",
            "best": {"segm": {"AP": 80.0}, "bbox": {"AP": 70.0}},
            "last": {"segm": {"AP": 78.0}, "bbox": {"AP": 68.0}},
            "inference": {"throughput_fps": 5.0},
        },
        "model_b": {
            "status": "ok",
            "best": {"segm": {"AP": 80.0}, "bbox": {"AP": 70.0}},
            "last": {"segm": {"AP": 79.0}, "bbox": {"AP": 68.0}},
            "inference": {"throughput_fps": 4.0},
        },
        "model_c": {
            "status": "ok",
            "best": {"segm": {"AP": 80.0}, "bbox": {"AP": 70.0}},
            "last": {"segm": {"AP": 79.0}, "bbox": {"AP": 68.0}},
            "inference": {"throughput_fps": 6.0},
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    out_json = tmp_path / "extended.json"
    out_csv = tmp_path / "extended.csv"
    out_md = tmp_path / "extended.md"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary",
            str(summary_path),
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
    assert [row["model_id"] for row in rows[:3]] == ["model_c", "model_b", "model_a"]
