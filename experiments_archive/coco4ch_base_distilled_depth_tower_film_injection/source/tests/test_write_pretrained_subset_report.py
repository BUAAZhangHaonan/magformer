from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_write_pretrained_subset_report_generates_markdown(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "write_pretrained_subset_report.py"
    summary = {
        "experiment": "demo",
        "output_root": str(tmp_path / "out"),
        "magformer_depthnorm_on": {"best": {"segm": {"AP": 77.5}}},
        "mgm_mask2former_depthnorm_on": {"best": {"segm": {"AP": 77.3}}},
        "official_mask2former_pretrained": {"best": {"segm": {"AP": 73.0}}},
        "maskrcnn_pretrained": {"best": {"segm": {"AP": 62.1}}},
        "yolov8_seg_pretrained": {"best": {"segm": {"AP": 51.6}}},
        "mgm_mask2former_nodpth_ref": {"best": {"segm": {"AP": 54.6}}},
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    out_path = tmp_path / "report.md"

    subprocess.run(
        [sys.executable, str(script), "--summary", str(summary_path), "--output", str(out_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    text = out_path.read_text(encoding="utf-8")
    assert "Pretrained Subset Report" in text
    assert "mgm_mask2former_nodpth_ref" in text
    assert "not directly comparable" in text
