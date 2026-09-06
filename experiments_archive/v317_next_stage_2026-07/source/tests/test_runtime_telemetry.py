from __future__ import annotations

import json
from pathlib import Path


def test_runtime_telemetry_writes_events_and_summary(tmp_path: Path) -> None:
    from baselines.runtime_telemetry import RuntimeTelemetry

    telemetry = RuntimeTelemetry(tmp_path, run_name="unit")
    telemetry.log_event("train_step", {"batch": 2, "data_time_sec": 0.25, "compute_time_sec": 0.5})
    telemetry.log_event("eval", {"images": 3, "postprocess_time_sec": 0.75})
    summary_path = telemetry.write_summary({"status": "ok"})

    events = [
        json.loads(line)
        for line in (tmp_path / "runtime_telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert [event["stage"] for event in events] == ["train_step", "eval"]
    assert events[0]["run_name"] == "unit"
    assert events[0]["metrics"]["batch"] == 2
    assert "rss_mb" in events[0]["resources"]
    assert summary["run_name"] == "unit"
    assert summary["status"] == "ok"
    assert summary["totals"]["data_time_sec"] == 0.25
    assert summary["totals"]["compute_time_sec"] == 0.5
    assert summary["totals"]["postprocess_time_sec"] == 0.75

