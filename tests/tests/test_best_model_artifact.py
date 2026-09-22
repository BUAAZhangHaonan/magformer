from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from magformer.engine.eval_runtime import EvaluationResult
from magformer.engine.trainer import Trainer
from magformer.engine.utils import load_checkpoint, load_torch_checkpoint
from tools.evaluate import apply_ema_weights_from_checkpoint
from tools.train import load_finetune_weights


class _StaticSampler:
    seed = 17


class _StaticStatefulLoader:
    batch_size = 1
    num_workers = 0
    prefetch_factor = None
    persistent_workers = False
    drop_last = False
    collate_fn = None

    def __init__(self) -> None:
        self.dataset = [0]
        self.sampler = _StaticSampler()
        self.cursor = 0

    def __iter__(self):
        return iter([{}])

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict) -> None:
        self.cursor = int(state_dict["cursor"])

    def validate_resume_state(self, state_dict, **kwargs):
        del kwargs
        return dict(state_dict)

    def is_epoch_exhausted(self) -> bool:
        return True

    def start_next_epoch(self, epoch: int):
        del epoch
        return iter(self)


def _trainer(tmp_path: Path, *, ema_enabled: bool) -> Trainer:
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runtime = {
        "grad_accum_steps": 1,
        "early_stop": {"enabled": False},
        "ema_enabled": ema_enabled,
    }
    if ema_enabled:
        runtime.update(ema_decay=0.9, ema_warmup_iters=1)
    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        train_loader=_StaticStatefulLoader(),
        val_loader=[{}],
        val_dataset=SimpleNamespace(category_ids=[], coco=None),
        config={"runtime": runtime},
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=2,
        eval_period=1,
        checkpoint_period=1,
        log_period=100,
        amp_enabled=False,
        clip_gradients=False,
        logger_config={"type": "none"},
    )
    trainer.current_iter = 1
    trainer.optimizer_step = 1
    return trainer


def _write_best_artifact(tmp_path: Path, monkeypatch):
    trainer = _trainer(tmp_path, ema_enabled=True)
    assert trainer.ema is not None
    trainer.ema.shadow["weight"].fill_(5.0)
    trainer.ema.shadow["bias"].fill_(4.0)

    def _fake_evaluation(model, *args, **kwargs):
        del args, kwargs
        assert model.weight.item() == pytest.approx(5.0)
        assert model.bias.item() == pytest.approx(4.0)
        return EvaluationResult(
            log_dict={"val/mAP": 0.8, "val/segm_AP": 0.8},
            coco_metrics={},
            coco_results_path=None,
            visualization_batch=None,
            visualization_outputs=None,
        )

    monkeypatch.setattr(
        "magformer.engine.trainer.run_inference_evaluation",
        _fake_evaluation,
    )
    metrics = trainer.evaluate()
    best_path = tmp_path / "best.pt"
    assert not (tmp_path / "model_best.pth").exists()
    artifact = load_torch_checkpoint(best_path, map_location="cpu")
    return trainer, metrics, best_path, artifact


def _write_raw_best_artifact(tmp_path: Path, monkeypatch):
    trainer = _trainer(tmp_path, ema_enabled=False)

    def _fake_evaluation(model, *args, **kwargs):
        del args, kwargs
        assert model.weight.item() == pytest.approx(2.0)
        assert model.bias.item() == pytest.approx(1.0)
        return EvaluationResult(
            log_dict={"val/mAP": 0.7, "val/segm_AP": 0.7},
            coco_metrics={},
            coco_results_path=None,
            visualization_batch=None,
            visualization_outputs=None,
        )

    monkeypatch.setattr(
        "magformer.engine.trainer.run_inference_evaluation",
        _fake_evaluation,
    )
    metrics = trainer.evaluate()
    best_path = tmp_path / "best.pt"
    artifact = load_torch_checkpoint(best_path, map_location="cpu")
    return trainer, metrics, best_path, artifact


def test_ema_best_artifact_primary_exactly_matches_evaluated_weights(
    tmp_path: Path, monkeypatch
) -> None:
    trainer, metrics, _, artifact = _write_best_artifact(tmp_path, monkeypatch)

    assert metrics["val/mAP"] == pytest.approx(0.8)
    assert trainer.model.weight.item() == pytest.approx(2.0)
    assert trainer.model.bias.item() == pytest.approx(1.0)
    assert trainer.model.training is True
    assert artifact["artifact_kind"] == "model_best"
    assert artifact["artifact_format_version"] == 2
    assert artifact["evaluated_weight_source"] == "ema"
    assert artifact["best_metric"] == pytest.approx(0.8)
    assert artifact["metrics"]["val/mAP"] == pytest.approx(0.8)
    assert artifact["model_state_dict"]["weight"].item() == pytest.approx(5.0)
    assert artifact["model_state_dict"]["bias"].item() == pytest.approx(4.0)
    assert artifact["raw_model_state_dict"]["weight"].item() == pytest.approx(2.0)
    assert artifact["raw_model_state_dict"]["bias"].item() == pytest.approx(1.0)
    optimizer_weight = trainer.optimizer.param_groups[0]["params"][0].item()
    assert artifact["raw_model_state_dict"]["weight"].item() == pytest.approx(
        optimizer_weight
    )
    for name, primary_value in artifact["model_state_dict"].items():
        assert torch.equal(primary_value, artifact["ema_state_dict"]["shadow"][name])
    for name, raw_value in artifact["raw_model_state_dict"].items():
        assert torch.equal(raw_value, trainer.model.state_dict()[name])
    for forbidden_key in (
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "scaler_state_dict",
        "checkpoint_format_version",
    ):
        assert forbidden_key not in artifact


def test_default_evaluate_and_warm_start_load_ema_best_primary(
    tmp_path: Path, monkeypatch
) -> None:
    _, _, best_path, _ = _write_best_artifact(tmp_path / "source", monkeypatch)

    warm_start_model = torch.nn.Linear(1, 1)
    load_finetune_weights(warm_start_model, str(best_path), strict=True)
    assert warm_start_model.weight.item() == pytest.approx(5.0)
    assert warm_start_model.bias.item() == pytest.approx(4.0)

    eval_model = torch.nn.Linear(1, 1)
    checkpoint, _ = load_checkpoint(str(best_path), eval_model, strict=True)
    assert eval_model.weight.item() == pytest.approx(5.0)
    assert eval_model.bias.item() == pytest.approx(4.0)
    assert eval_model(torch.ones(1, 1)).item() == pytest.approx(9.0)
    apply_ema_weights_from_checkpoint(eval_model, checkpoint)
    assert eval_model.weight.item() == pytest.approx(5.0)
    assert eval_model.bias.item() == pytest.approx(4.0)


def test_raw_best_artifact_primary_and_default_evaluate_match_raw_weights(
    tmp_path: Path, monkeypatch
) -> None:
    trainer, metrics, best_path, artifact = _write_raw_best_artifact(
        tmp_path, monkeypatch
    )

    assert metrics["val/mAP"] == pytest.approx(0.7)
    assert artifact["artifact_format_version"] == 2
    assert artifact["evaluated_weight_source"] == "raw"
    assert "ema_state_dict" not in artifact
    for name, primary_value in artifact["model_state_dict"].items():
        assert torch.equal(primary_value, artifact["raw_model_state_dict"][name])
        assert torch.equal(primary_value, trainer.model.state_dict()[name])

    eval_model = torch.nn.Linear(1, 1)
    load_checkpoint(str(best_path), eval_model, strict=True)
    assert eval_model.weight.item() == pytest.approx(2.0)
    assert eval_model.bias.item() == pytest.approx(1.0)
    assert eval_model(torch.ones(1, 1)).item() == pytest.approx(3.0)


def test_best_artifact_claimed_ema_without_ema_state_fails_loudly(
    tmp_path: Path, monkeypatch
) -> None:
    _, _, _, artifact = _write_best_artifact(tmp_path / "source", monkeypatch)
    artifact.pop("ema_state_dict")
    invalid_path = tmp_path / "missing_ema.pth"
    torch.save(artifact, invalid_path)

    with pytest.raises(RuntimeError, match="claims.*ema.*no valid ema_state"):
        load_checkpoint(str(invalid_path), torch.nn.Linear(1, 1), strict=True)


def test_best_artifact_primary_metadata_mismatch_fails_loudly(
    tmp_path: Path, monkeypatch
) -> None:
    _, _, _, artifact = _write_best_artifact(tmp_path / "source", monkeypatch)
    artifact["model_state_dict"]["weight"] = torch.tensor([[99.0]])
    invalid_path = tmp_path / "mismatched_primary.pth"
    torch.save(artifact, invalid_path)

    with pytest.raises(RuntimeError, match="state mismatch"):
        load_checkpoint(str(invalid_path), torch.nn.Linear(1, 1), strict=True)


def test_best_artifact_without_weight_source_is_rejected_as_ambiguous(
    tmp_path: Path, monkeypatch
) -> None:
    _, _, _, artifact = _write_raw_best_artifact(tmp_path / "source", monkeypatch)
    artifact.pop("evaluated_weight_source")
    invalid_path = tmp_path / "ambiguous.pth"
    torch.save(artifact, invalid_path)

    with pytest.raises(RuntimeError, match="evaluated_weight_source"):
        load_checkpoint(str(invalid_path), torch.nn.Linear(1, 1), strict=True)


def test_best_pt_is_rejected_for_full_state_resume(
    tmp_path: Path, monkeypatch
) -> None:
    _, _, best_path, _ = _write_best_artifact(tmp_path / "source", monkeypatch)
    destination = _trainer(tmp_path / "destination", ema_enabled=False)
    weight_before = destination.model.weight.detach().clone()

    with pytest.raises(ValueError, match="named exactly last.pt"):
        destination.resume(str(best_path))

    assert torch.equal(destination.model.weight.detach(), weight_before)


def test_periodic_checkpoint_is_full_training_state_and_resumes(
    tmp_path: Path,
) -> None:
    source = _trainer(tmp_path / "source", ema_enabled=False)
    source.save_checkpoint()
    checkpoint_path = tmp_path / "source" / "last.pt"
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")

    assert checkpoint["artifact_kind"] == "training_state"
    assert checkpoint["checkpoint_format_version"] == 3
    assert "resume_contract" in checkpoint
    assert len(checkpoint["rank_states"]) == 1
    assert "optimizer_state_dict" in checkpoint
    assert "model_state_dict" in checkpoint

    destination = _trainer(tmp_path / "destination", ema_enabled=False)
    with torch.no_grad():
        destination.model.weight.zero_()
    destination.current_iter = 0
    destination.optimizer_step = 0
    destination.resume(str(checkpoint_path))

    assert destination.current_iter == 1
    assert destination.optimizer_step == 1
    assert destination.model.weight.item() == pytest.approx(2.0)

    assert sorted(path.name for path in (tmp_path / "source").glob("*.pt")) == [
        "last.pt"
    ]


def test_full_state_resume_rejects_old_checkpoint_filename(tmp_path: Path) -> None:
    source = _trainer(tmp_path / "source", ema_enabled=False)
    source.save_checkpoint()
    old_path = tmp_path / "source" / "checkpoint_iter_0000001.pth"
    (tmp_path / "source" / "last.pt").rename(old_path)

    destination = _trainer(tmp_path / "destination", ema_enabled=False)
    with pytest.raises(ValueError, match="named exactly last.pt"):
        destination.resume(str(old_path))
