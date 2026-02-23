from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_summarize_20ep_scratch8_schema(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "summarize_0831_1k_20ep_scratch8.py"
    assert script.exists()

    proc = subprocess.run(
        [sys.executable, str(script), "--output-root", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(proc.stdout)

    assert summary["experiment"] == "0831_1k_20ep_scratch8"
    assert summary["output_root"] == str(tmp_path)
    for k in [
        "magformer_scratch",
        "mgm_mask2former_scratch",
        "msmformer_scratch",
        "uoais_scratch",
        "ucn_scratch",
        "official_mask2former_scratch",
        "maskrcnn_scratch",
        "yolov8_seg_scratch",
    ]:
        assert k in summary
        assert "status" in summary[k]
