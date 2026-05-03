from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_summarize_suite_reads_peak_memory_file(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "summarize_suite.py"
    out_root = tmp_path / "0831_1k_20ep_1024_lightdepth_stage_a"
    model_dir = out_root / "magformer_lightdepth_test"
    model_dir.mkdir(parents=True)
    (model_dir / "metrics.cocoeval.json").write_text(
        json.dumps({"iteration": 10, "segm/AP": 12.3, "segm/AP50": 30.0, "segm/AP75": 10.0}),
        encoding="utf-8",
    )
    (model_dir / "metrics_log.jsonl").write_text(
        json.dumps({"phase": "val", "iter": 10, "val/segm_AP": 0.123}) + "\n",
        encoding="utf-8",
    )
    (model_dir / "peak_memory_mb.txt").write_text("2048.5\n", encoding="utf-8")

    summary_name = "summary_peak.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-root",
            str(out_root),
            "--write",
            "--write-name",
            summary_name,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((out_root / summary_name).read_text(encoding="utf-8"))
    assert float(payload["magformer_lightdepth_test"]["peak_memory_mb"]) == 2048.5
