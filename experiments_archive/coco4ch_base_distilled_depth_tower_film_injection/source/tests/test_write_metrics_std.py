from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def test_write_metrics_std_from_magformer_metrics_log(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "write_metrics_std.py"
    assert script.exists()

    (tmp_path / "metrics_log.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "iter": 0,
                        "phase": "train",
                        "wall_time_iso": "2026-02-24T00:00:00-00:00",
                        "elapsed_sec": 0.1,
                        "train/loss": 1.0,
                        "train/lr": 0.001,
                    }
                ),
                json.dumps(
                    {
                        "iter": 10,
                        "phase": "val",
                        "wall_time_iso": "2026-02-24T00:01:00-00:00",
                        "elapsed_sec": 60.0,
                        # MAGFormer evaluator returns 0..1; std writer must convert to 0..100.
                        "val/segm_AP": 0.5,
                        "val/segm_AP50": 0.75,
                        "val/bbox_AP": 0.25,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, str(script), "--out-dir", str(tmp_path), "--iters-per-epoch", "10"],
        check=True,
        capture_output=True,
        text=True,
    )

    out_jsonl = tmp_path / "metrics_std.jsonl"
    out_csv = tmp_path / "metrics_std.csv"
    assert out_jsonl.exists()
    assert out_csv.exists()

    rows = _read_jsonl(out_jsonl)
    assert len(rows) == 2

    train_row, val_row = rows
    assert train_row["phase"] == "train"
    assert train_row["step"] == 0
    assert train_row["step_unit"] == "iter"
    assert train_row["train/loss"] == 1.0

    assert val_row["phase"] == "val"
    assert val_row["step"] == 10
    assert val_row["val/segm_AP"] == 50.0
    assert val_row["val/segm_AP50"] == 75.0
    assert val_row["val/bbox_AP"] == 25.0


def test_write_metrics_std_from_detectron2_metrics_json(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "write_metrics_std.py"
    assert script.exists()

    (tmp_path / "metrics.json").write_text(
        "\n".join(
            [
                json.dumps({"iteration": 0, "total_loss": 1.234, "lr": 0.001}),
                json.dumps({"iteration": 9, "segm/AP": 12.3, "segm/AP50": 34.5, "bbox/AP": 7.8}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, str(script), "--out-dir", str(tmp_path), "--iters-per-epoch", "10"],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = _read_jsonl(tmp_path / "metrics_std.jsonl")
    assert len(rows) == 2

    train_row, val_row = rows
    assert train_row["phase"] == "train"
    assert train_row["step"] == 0
    assert train_row["train/loss"] == 1.234
    assert train_row["train/lr"] == 0.001

    assert val_row["phase"] == "val"
    assert val_row["step"] == 9
    assert val_row["val/segm_AP"] == 12.3
    assert val_row["val/segm_AP50"] == 34.5
    assert val_row["val/bbox_AP"] == 7.8

