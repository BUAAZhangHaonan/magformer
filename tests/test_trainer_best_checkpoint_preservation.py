import torch

from magformer.engine.trainer import Trainer


def test_train_does_not_mark_final_checkpoint_as_best(monkeypatch, tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        model=model,
        criterion=None,
        optimizer=optimizer,
        train_loader=[{"images": torch.zeros(1, 1), "depths": torch.zeros(1, 1)}],
        val_loader=None,
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

    def _fake_save_checkpoint(is_best=False):
        calls.append(bool(is_best))

    monkeypatch.setattr(trainer, "_train_step", _fake_train_step)
    monkeypatch.setattr(trainer, "save_checkpoint", _fake_save_checkpoint)

    trainer.train()

    # Final checkpoint must be saved as regular checkpoint, not best.
    assert calls
    assert calls[-1] is False
