from types import SimpleNamespace
import torch

from tools.train import build_lr_scheduler


def _cfg():
    solver = SimpleNamespace(
        lr_scheduler="multistep",
        max_iter=2000,
        warmup_iters=200,
        warmup_factor=0.01,
        warmup_method="linear",
        steps=[1500, 1800],
        gamma=0.1,
    )
    return SimpleNamespace(solver=solver)


def test_multistep_scheduler_uses_warmup_then_milestones():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=1e-4)
    scheduler = build_lr_scheduler(optimizer, _cfg())

    lrs = []
    for _ in range(2001):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # warmup should be active in early steps
    assert lrs[0] < 1e-4
    assert lrs[100] < 1e-4

    # before milestones, should reach base lr
    assert abs(lrs[300] - 1e-4) < 1e-9

    # after first milestone (1500), lr should drop by gamma
    assert abs(lrs[1500] - 1e-5) < 1e-9

    # after second milestone (1800), lr should drop again
    assert abs(lrs[1800] - 1e-6) < 1e-9
