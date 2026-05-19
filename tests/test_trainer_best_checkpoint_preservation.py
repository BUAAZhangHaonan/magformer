import pytest
import torch

from magformer.engine.trainer import Trainer


def _make_trainer(tmp_path, *, eval_period, checkpoint_period):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    return Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        train_loader=[{"images": torch.zeros(1, 1), "depths": torch.zeros(1, 1)}],
        val_loader=None,
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        max_iter=1,
        eval_period=eval_period,
        checkpoint_period=checkpoint_period,
        log_period=1,
        amp_enabled=False,
    )


def test_train_does_not_mark_final_checkpoint_as_best(monkeypatch, tmp_path):
    trainer = _make_trainer(tmp_path, eval_period=100, checkpoint_period=100)

    calls = []

    def _fake_train_step(_batch):
        trainer.current_iter += 1
        return {"total_loss": torch.tensor(1.0)}

    def _fake_save_checkpoint(is_best=False):
        calls.append(bool(is_best))

    monkeypatch.setattr(trainer, "_train_step", _fake_train_step)
    monkeypatch.setattr(trainer, "save_checkpoint", _fake_save_checkpoint)

    trainer.train()

    # Final checkpoint must be saved as regular checkpoint, not best.
    assert calls
    assert calls[-1] is False


def test_checkpoint_boundary_saves_before_eval_exception(monkeypatch, tmp_path):
    trainer = _make_trainer(tmp_path, eval_period=1, checkpoint_period=1)
    events = []

    def _fake_train_step(_batch):
        trainer.current_iter += 1
        return {"total_loss": torch.tensor(1.0)}

    def _fake_save_checkpoint(is_best=False):
        events.append(("save", trainer.current_iter, bool(is_best)))
        (tmp_path / f"checkpoint_iter_{trainer.current_iter:07d}.pth").write_text(
            "saved\n", encoding="utf-8"
        )

    def _failing_evaluate():
        events.append(("eval", trainer.current_iter, None))
        raise RuntimeError("eval failed")

    monkeypatch.setattr(trainer, "_train_step", _fake_train_step)
    monkeypatch.setattr(trainer, "save_checkpoint", _fake_save_checkpoint)
    monkeypatch.setattr(trainer, "evaluate", _failing_evaluate)

    with pytest.raises(RuntimeError, match="eval failed"):
        trainer.train()

    assert events == [("save", 1, False), ("eval", 1, None)]
    assert (tmp_path / "checkpoint_iter_0000001.pth").exists()


def test_final_checkpoint_saves_before_final_eval_exception(monkeypatch, tmp_path):
    trainer = _make_trainer(tmp_path, eval_period=100, checkpoint_period=100)
    events = []

    def _fake_train_step(_batch):
        trainer.current_iter += 1
        return {"total_loss": torch.tensor(1.0)}

    def _fake_save_checkpoint(is_best=False):
        events.append(("save", trainer.current_iter, bool(is_best)))
        (tmp_path / f"checkpoint_iter_{trainer.current_iter:07d}.pth").write_text(
            "saved\n", encoding="utf-8"
        )

    def _failing_evaluate():
        events.append(("eval", trainer.current_iter, None))
        raise RuntimeError("final eval failed")

    monkeypatch.setattr(trainer, "_train_step", _fake_train_step)
    monkeypatch.setattr(trainer, "save_checkpoint", _fake_save_checkpoint)
    monkeypatch.setattr(trainer, "evaluate", _failing_evaluate)

    with pytest.raises(RuntimeError, match="final eval failed"):
        trainer.train()

    assert events == [("save", 1, False), ("eval", 1, None)]
    assert (tmp_path / "checkpoint_iter_0000001.pth").exists()
