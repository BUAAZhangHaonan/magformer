from __future__ import annotations

import torch

from scripts.analysis.benchmark_inference import _measure_latency_phased


def test_measure_latency_phased_reports_phase_breakdown() -> None:
    items = [1, 2, 3, 4]

    def infer_fn(item: int) -> dict:
        return {"value": item}

    def postprocess_fn(result: dict) -> dict:
        return {"value": result["value"] + 1}

    def export_fn(result: dict) -> dict:
        return {"value": result["value"] * 2}

    metrics = _measure_latency_phased(
        items=items,
        infer_fn=infer_fn,
        postprocess_fn=postprocess_fn,
        export_fn=export_fn,
        device=torch.device("cpu"),
        warmup=1,
        timed_images=2,
    )

    assert metrics["timed_images"] == 2
    assert "latency_ms_mean" in metrics
    assert "model_forward_latency_ms_mean" in metrics
    assert "postprocess_latency_ms_mean" in metrics
    assert "export_latency_ms_mean" in metrics
