import json

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

    for key in ["wall_time", "wall_time_iso", "elapsed_sec", "iter_time_sec", "eta_sec"]:
        assert key in payload
    assert payload["phase"] == "train"
    assert isinstance(payload["wall_time"], float)
    assert isinstance(payload["wall_time_iso"], str)
