from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires 2 CUDA GPUs",
)


def test_magformer_real_ddp_smoke_runs_eval_and_config_weight_eval(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260321_ddp_smoke_canary.sh"
    out_root = tmp_path / "ddp_smoke"

    subprocess.run(
        [
            "bash",
            str(script),
            "--output-root",
            str(out_root),
            "--magformer-only",
            "--run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    train_out = out_root / "magformer_ddp_nodpth_ref"
    eval_out = out_root / "magformer_eval_from_config"

    assert (train_out / "model_best.pth").exists()
    assert (train_out / "checkpoint_iter_0000001.pth").exists()
    assert (train_out / "metrics.cocoeval.json").exists()
    assert (train_out / "coco_instances_results.json").exists()
    assert (eval_out / "metrics.cocoeval.json").exists()
    assert (eval_out / "coco_instances_results.json").exists()

    train_metrics = json.loads((train_out / "metrics.cocoeval.json").read_text(encoding="utf-8"))
    train_preds = json.loads((train_out / "coco_instances_results.json").read_text(encoding="utf-8"))
    eval_metrics = json.loads((eval_out / "metrics.cocoeval.json").read_text(encoding="utf-8"))
    eval_preds = json.loads((eval_out / "coco_instances_results.json").read_text(encoding="utf-8"))
    metrics_log_lines = [
        json.loads(line)
        for line in (train_out / "metrics_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    val_rows = [row for row in metrics_log_lines if row.get("phase") == "val"]
    assert val_rows
    last_val = val_rows[-1]

    assert "segm/AP" in train_metrics
    assert "segm/AP" in eval_metrics
    assert float(last_val["val/diag_num_eval_images"]) == 4.0
    assert len(train_preds) > 0
    assert len(eval_preds) > 0
    assert "val/loss" not in json.dumps(last_val)
