from types import SimpleNamespace
import pytest
import torch

from tools.train import build_lr_scheduler


def _cfg(grad_accum_steps=1, iteration_unit="optimizer_step"):
    solver = SimpleNamespace(
        iteration_unit=iteration_unit,
        lr_scheduler="multistep",
        max_iter=2000,
        warmup_iters=200,
        warmup_factor=0.01,
        warmup_method="linear",
        steps=[1500, 1800],
        gamma=0.1,
    )
    runtime = SimpleNamespace(grad_accum_steps=grad_accum_steps)
    return SimpleNamespace(solver=solver, runtime=runtime)


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


def test_optimizer_step_schedule_is_not_scaled_by_gradient_accumulation():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=1e-4)
    scheduler = build_lr_scheduler(optimizer, _cfg(grad_accum_steps=4))

    lrs = []
    for _ in range(1801):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] < 1e-4
    assert lrs[100] < 1e-4
    assert abs(lrs[300] - 1e-4) < 1e-9
    assert abs(lrs[1500] - 1e-5) < 1e-9
    assert abs(lrs[1800] - 1e-6) < 1e-9


def test_explicit_legacy_schedule_converts_micro_steps():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=1e-4)
    scheduler = build_lr_scheduler(
        optimizer,
        _cfg(grad_accum_steps=4, iteration_unit="legacy_micro_step"),
    )

    lrs = []
    for _ in range(501):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] < 1e-4
    assert abs(lrs[75] - 1e-4) < 1e-9
    assert abs(lrs[375] - 1e-5) < 1e-9
    assert abs(lrs[450] - 1e-6) < 1e-9


def test_scheduler_rejects_non_divisible_micro_step_schedule():
    cfg = _cfg(grad_accum_steps=4, iteration_unit="legacy_micro_step")
    cfg.solver.warmup_iters = 201
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=1e-4)

    with pytest.raises(ValueError, match="solver.warmup_iters=201"):
        build_lr_scheduler(optimizer, cfg)
