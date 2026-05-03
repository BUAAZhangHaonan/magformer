from __future__ import annotations

from tools.train import resolve_distributed_context


def test_resolve_distributed_context_uses_torchrun_env_for_rank_and_device() -> None:
    env = {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"}
    ctx = resolve_distributed_context(
        ddp_enabled=True,
        runtime_gpus=[0, 1],
        runtime_device="cuda",
        env=env,
    )

    assert ctx["is_distributed"] is True
    assert ctx["world_size"] == 2
    assert ctx["rank"] == 1
    assert ctx["local_rank"] == 1
    assert ctx["device_index"] == 1


def test_resolve_distributed_context_rejects_multi_gpu_ddp_without_torchrun_env() -> None:
    ctx = resolve_distributed_context(
        ddp_enabled=True,
        runtime_gpus=[0, 1],
        runtime_device="cuda",
        env={},
    )

    assert ctx["is_distributed"] is False
    assert ctx["requires_launcher"] is True
