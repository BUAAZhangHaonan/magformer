from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_external_artifacts(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.cocoeval.json").write_text(
        json.dumps(
            {
                "iteration": 20,
                "segm/AP": 1.0,
                "segm/AP50": 2.0,
                "segm/AP75": 3.0,
                "bbox/AP": 4.0,
                "bbox/AP50": 5.0,
                "bbox/AP75": 6.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (model_dir / "coco_instances_results.json").write_text("[]\n", encoding="utf-8")
    (model_dir / "metadata.json").write_text(
        json.dumps({"model_id": model_dir.name}) + "\n",
        encoding="utf-8",
    )


def test_summarize_suite_surfaces_external_baseline_fidelity(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "summarize_suite.py"
    out_root = tmp_path / "suite"

    for model_id in ["cellpose", "iaunet", "stardist"]:
        _write_external_artifacts(out_root / model_id)

    summary_name = "summary_suite.json"
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

    summary = json.loads((out_root / summary_name).read_text(encoding="utf-8"))
    assert summary["cellpose"]["implementation_fidelity"]["implementation_kind"] == "custom-like"
    assert summary["iaunet"]["implementation_fidelity"]["implementation_kind"] == "paper-inspired-custom"
    assert summary["stardist"]["implementation_fidelity"]["implementation_kind"] == "official-library"
