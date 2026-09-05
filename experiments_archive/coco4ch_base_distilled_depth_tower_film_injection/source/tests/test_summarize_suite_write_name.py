from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_summarize_suite_can_write_noncanonical_summary_name(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "summarize_suite.py"
    out_root = tmp_path / "0831_1k_20ep_trackp"
    model_dir = out_root / "magformer"
    model_dir.mkdir(parents=True)
    (model_dir / "metrics.cocoeval.json").write_text(
        json.dumps({"iteration": 10, "segm/AP": 12.3, "segm/AP50": 30.0, "segm/AP75": 10.0}),
        encoding="utf-8",
    )
    (model_dir / "metrics_log.jsonl").write_text(
        json.dumps({"phase": "val", "iter": 10, "val/segm_AP": 0.123}) + "\n",
        encoding="utf-8",
    )
    custom_name = "summary_0831_1k_20ep_trackp_raw.json"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-root",
            str(out_root),
            "--write",
            "--write-name",
            custom_name,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert (out_root / custom_name).exists()
