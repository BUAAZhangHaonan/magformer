from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_write_run_metadata_start_and_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "write_run_metadata.py"
    assert script.exists()

    (tmp_path / "wall_time_sec.txt").write_text("12.5\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--phase",
            "start",
            "--out-dir",
            str(tmp_path),
            "--track",
            "trackp",
            "--register",
            "0909",
            "--dataset-root",
            "/tmp/ecc0909",
            "--model-id",
            "magformer",
            "--candidate-id",
            "C1",
            "--run-tag",
            "final",
            "--command",
            "bash run.sh --foo bar",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    meta_path = tmp_path / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["track"] == "trackp"
    assert meta["register"] == "0909"
    assert meta["model_id"] == "magformer"
    assert meta["candidate_id"] == "C1"
    assert meta["run_tag"] == "final"
    assert isinstance(meta.get("start_time_iso"), str)

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--phase",
            "end",
            "--out-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    meta2 = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta2["track"] == "trackp"
    assert isinstance(meta2.get("end_time_iso"), str)
    assert meta2.get("wall_time_sec") == 12.5

