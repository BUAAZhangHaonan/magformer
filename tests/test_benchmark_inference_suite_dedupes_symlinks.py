from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_inference_suite_dedupes_realpaths(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "benchmark_inference_suite.py"
    out_root = tmp_path / "suite"
    target = out_root / "model_a"
    target.mkdir(parents=True)
    link = out_root / "model_b"
    link.symlink_to(target, target_is_directory=True)
    summary = {
        "experiment": "x",
        "output_root": str(out_root),
        "model_a": {"status": "ok"},
        "model_b": {"status": "ok"},
    }
    summary_path = out_root / "summary_suite.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    res = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-root",
            str(out_root),
            "--dataset-root",
            str(tmp_path),
            "--summary",
            str(summary_path),
            "--warmup",
            "1",
            "--timed-images",
            "1",
            "--continue-on-error",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert res.stdout.count("[benchmark-suite]") == 1
