from __future__ import annotations

import json
import os
import resource
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage > 1024 * 1024 * 1024:
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def _first_visible_gpu() -> str | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible == "-1":
        return None
    return visible.split(",", 1)[0].strip() or None


def _nvidia_smi_snapshot() -> Dict[str, float | int] | None:
    gpu_id = _first_visible_gpu()
    if gpu_id is None:
        return None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 4:
        return None
    try:
        return {
            "gpu_util_percent": int(parts[0]),
            "gpu_mem_util_percent": int(parts[1]),
            "gpu_memory_used_mb": int(parts[2]),
            "gpu_memory_total_mb": int(parts[3]),
        }
    except ValueError:
        return None


def snapshot_resources() -> Dict[str, Any]:
    resources: Dict[str, Any] = {"rss_mb": _rss_mb()}
    gpu = _nvidia_smi_snapshot()
    if gpu is not None:
        resources.update(gpu)
    return resources


class RuntimeTelemetry:
    def __init__(self, output_dir: str | Path, *, run_name: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = str(run_name)
        self.start_time = time.perf_counter()
        self.events_path = self.output_dir / "runtime_telemetry.jsonl"
        self.summary_path = self.output_dir / "runtime_telemetry_summary.json"
        self.totals: Dict[str, float] = {}

    def _update_totals(self, metrics: Mapping[str, Any]) -> None:
        for key, value in metrics.items():
            if not key.endswith("_time_sec"):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            self.totals[key] = self.totals.get(key, 0.0) + numeric

    def log_event(self, stage: str, metrics: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        payload_metrics = dict(metrics or {})
        self._update_totals(payload_metrics)
        event = {
            "time_iso": _now_iso(),
            "elapsed_sec": max(0.0, time.perf_counter() - self.start_time),
            "run_name": self.run_name,
            "stage": str(stage),
            "metrics": payload_metrics,
            "resources": snapshot_resources(),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return event

    @contextmanager
    def stage(self, stage: str, **metrics: Any) -> Iterator[Dict[str, Any]]:
        start = time.perf_counter()
        payload = dict(metrics)
        try:
            yield payload
        finally:
            payload[f"{stage}_time_sec"] = max(0.0, time.perf_counter() - start)
            self.log_event(stage, payload)

    def write_summary(self, extra: Mapping[str, Any] | None = None) -> Path:
        summary = {
            "time_iso": _now_iso(),
            "run_name": self.run_name,
            "elapsed_sec": max(0.0, time.perf_counter() - self.start_time),
            "totals": dict(sorted(self.totals.items())),
            "resources": snapshot_resources(),
        }
        summary.update(dict(extra or {}))
        self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.summary_path

