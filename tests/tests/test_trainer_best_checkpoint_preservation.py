import pytest
import torch

from magformer.engine.trainer import Trainer


class _OneBatchLoader:
    def __init__(self):
        self.batch = {"images": torch.zeros(1, 1), "depths": torch.zeros(1, 1)}

    def __iter__(self):
        return iter([self.batch])

    def is_epoch_exhausted(self) -> bool:
        return True

    def start_next_epoch(self, epoch: int):
        del epoch
        return iter(self)


def test_train_does_not_mark_final_checkpoint_as_best(monkeypatch, tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        train_loader=_OneBatchLoader(),
        val_loader=None,
        config={"solver": {"iteration_unit": "legacy_micro_step"}},
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=1,
        eval_period=100,
        checkpoint_period=100,
        log_period=1,
        amp_enabled=False,
    )

    calls = []

    def _fake_train_step(_batch):
        trainer.current_iter += 1
        return {"total_loss": torch.tensor(1.0)}

    def _fake_save_checkpoint():
        calls.append("last")

    monkeypatch.setattr(trainer, "_train_step", _fake_train_step)
    monkeypatch.setattr(trainer, "save_checkpoint", _fake_save_checkpoint)

    trainer.train()

    # Final training state uses the single resumable checkpoint path.
    assert calls
    assert calls[-1] == "last"


def test_eval_milestone_saves_last_before_eval_error(monkeypatch, tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        train_loader=_OneBatchLoader(),
        val_loader=None,
        config={"solver": {"iteration_unit": "legacy_micro_step"}},
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=2,
        eval_period=1,
        checkpoint_period=100,
        log_period=1,
        amp_enabled=False,
    )

    calls = []

    def _fake_train_step(_batch):
        trainer.current_iter += 1
        return {"total_loss": torch.tensor(1.0)}

    def _fake_save_checkpoint():
        calls.append("last")

    def _failing_evaluate():
        calls.append("eval")
        raise RuntimeError("visualization failed")

    monkeypatch.setattr(trainer, "_train_step", _fake_train_step)
    monkeypatch.setattr(trainer, "save_checkpoint", _fake_save_checkpoint)
    monkeypatch.setattr(trainer, "evaluate", _failing_evaluate)

    with pytest.raises(RuntimeError, match="visualization failed"):
        trainer.train()

    assert calls == ["last", "eval"]
