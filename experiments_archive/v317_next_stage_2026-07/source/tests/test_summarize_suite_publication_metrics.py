from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_summarize_suite_exposes_publication_metrics_from_final_cocoeval_live_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "summarize_suite.py"
    out_root = repo_root / "output" / "experiments" / "20260318_1k_1566_20ep_1024_full19"

    result = subprocess.run(
        [sys.executable, str(script), "--output-root", str(out_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    mgm_final = json.loads(
        (out_root / "mgm_mask2former_depthnorm_on" / "metrics.cocoeval.json").read_text(encoding="utf-8")
    )
    mask2former_final = json.loads(
        (out_root / "mask2former" / "metrics.cocoeval.json").read_text(encoding="utf-8")
    )

    assert payload["mgm_mask2former_depthnorm_on"]["publication"]["bbox"]["AP"] == pytest.approx(
        float(mgm_final["bbox/AP"])
    )
    assert payload["mask2former"]["publication"]["bbox"]["AP"] == pytest.approx(
        float(mask2former_final["bbox/AP"])
    )
