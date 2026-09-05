import json

import pytest
import torch

from magformer.engine.trainer import Trainer


def test_metrics_log_jsonl_has_time_fields(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        model=model,
        criterion=torch.nn.MSELoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=10,
        eval_period=10,
        checkpoint_period=10,
        log_period=1,
        amp_enabled=False,
    )

    # Simulate a valid iter-time window so ETA is numeric.
    trainer._iter_time_window_sec.append(0.05)
    trainer._append_metrics_log({"train/loss": 1.0}, phase="train")

    line = (tmp_path / "metrics_log.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = json.loads(line)

    for key in [
        "micro_step",
        "optimizer_step",
        "amp_skipped_steps",
        "wall_time",
        "wall_time_iso",
        "elapsed_sec",
        "iter_time_sec",
        "eta_sec",
        "amp_scale",
        "amp_scale_before_step",
        "amp_scale_after_step",
        "preclip_grad_norm",
        "grad_finite",
        "first_nonfinite_grad_param",
        "nonfinite_grad_count",
        "consecutive_amp_skips",
        "cuda_memory_allocated_mb",
        "cuda_memory_reserved_mb",
        "cuda_max_memory_reserved_mb",
    ]:
        assert key in payload
    assert payload["iter"] == payload["micro_step"] == 0
    assert payload["optimizer_step"] == 0
    assert payload["phase"] == "train"
    assert isinstance(payload["wall_time"], float)
    assert isinstance(payload["wall_time_iso"], str)
    assert payload["amp_scale"] is None
    assert payload["amp_scale_before_step"] is None
    assert payload["amp_scale_after_step"] is None
    assert payload["preclip_grad_norm"] is None
    assert payload["grad_finite"] is None
    assert payload["first_nonfinite_grad_param"] is None
    assert payload["nonfinite_grad_count"] == 0
    assert payload["consecutive_amp_skips"] == 0
    assert payload["cuda_memory_allocated_mb"] is None
    assert payload["cuda_memory_reserved_mb"] is None
    assert payload["cuda_max_memory_reserved_mb"] is None


def test_metrics_log_rejects_non_finite_runtime_telemetry(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        criterion=torch.nn.MSELoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=1,
        eval_period=1,
        checkpoint_period=1,
        log_period=1,
        amp_enabled=False,
    )
    trainer._last_preclip_grad_norm = float("nan")

    with pytest.raises(FloatingPointError, match="preclip_grad_norm"):
        trainer._append_metrics_log({"train/loss": 1.0}, phase="train")
