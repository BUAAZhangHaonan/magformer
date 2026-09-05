from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from magformer.config.schema import RuntimeConfig
from tools import train as train_tool


def _runtime(device: str = "cuda", memory_fraction=None) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        memory_fraction=memory_fraction,
        cpu_threads=8,
        cpu_interop_threads=1,
    )


def test_runtime_resource_limit_schema_defaults_and_bounds() -> None:
    config = RuntimeConfig()

    assert config.memory_fraction is None
    assert config.cpu_threads == 8
    assert config.cpu_interop_threads == 1

    for invalid_fraction in (0.0, -0.1, 1.00001, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            RuntimeConfig(memory_fraction=invalid_fraction)
    with pytest.raises(ValidationError):
        RuntimeConfig(cpu_threads=0)
    with pytest.raises(ValidationError):
        RuntimeConfig(cpu_interop_threads=0)


def test_configure_cpu_runtime_applies_environment_and_torch_limits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, int]] = []
    applied = {"intra_op": 0, "inter_op": 0}

    def set_num_threads(value: int) -> None:
        calls.append(("intra_op", value))
        applied["intra_op"] = value

    def set_num_interop_threads(value: int) -> None:
        calls.append(("inter_op", value))
        applied["inter_op"] = value

    monkeypatch.setattr(torch, "set_num_threads", set_num_threads)
    monkeypatch.setattr(torch, "set_num_interop_threads", set_num_interop_threads)
    monkeypatch.setattr(torch, "get_num_threads", lambda: applied["intra_op"])
    monkeypatch.setattr(
        torch, "get_num_interop_threads", lambda: applied["inter_op"]
    )

    train_tool.configure_cpu_runtime(_runtime())

    assert calls == [("intra_op", 8), ("inter_op", 1)]
    for variable in train_tool._CPU_THREAD_ENV_VARS:
        assert train_tool.os.environ[variable] == "8"
    assert "intra_op=8, inter_op=1" in capsys.readouterr().out


def test_initialize_runtime_device_caps_cuda_before_ddp(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda index: calls.append(("set_device", index))
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, index: calls.append(("set_fraction", fraction, index)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=24_576 * 1024 * 1024),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(
            ("init_process_group", kwargs["backend"], kwargs["init_method"])
        ),
    )

    device = train_tool.initialize_runtime_device(
        _runtime(memory_fraction=0.85),
        {"device_index": 2, "is_distributed": True},
    )

    assert device == torch.device("cuda:2")
    assert calls == [
        ("set_device", 2),
        ("set_fraction", 0.85, 2),
        ("init_process_group", "nccl", "env://"),
    ]
    output = capsys.readouterr().out
    assert "fraction=0.8500" in output
    assert "cap_mib=20889.60" in output


def test_initialize_runtime_device_without_fraction_never_calls_allocator_api(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda index: calls.append(("set_device", index))
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda *args: calls.append(("set_fraction", *args)),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    device = train_tool.initialize_runtime_device(
        _runtime(memory_fraction=None),
        {"device_index": 4, "is_distributed": False},
    )

    assert device == torch.device("cuda:4")
    assert calls == [("set_device", 4)]
    assert "disabled (device=cuda:4)" in capsys.readouterr().out


def test_initialize_runtime_device_cpu_path_never_touches_cuda_or_ddp(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_call(*args, **kwargs):
        del args, kwargs
        raise AssertionError("CPU runtime must not initialize CUDA or DDP")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_call)
    monkeypatch.setattr(torch.cuda, "set_device", unexpected_call)
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", unexpected_call
    )
    monkeypatch.setattr(torch.distributed, "init_process_group", unexpected_call)

    device = train_tool.initialize_runtime_device(
        _runtime(device="cpu"),
        {"device_index": 0, "is_distributed": False},
    )

    assert device == torch.device("cpu")
    assert "disabled (device=cpu)" in capsys.readouterr().out
