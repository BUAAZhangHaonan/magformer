from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "baselines" / "run_unet_instance_ecc.py"
    spec = importlib.util.spec_from_file_location("unet_runner", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_recommend_main_process_threads_caps_thread_count_when_workers_enabled() -> None:
    mod = _load_module()
    assert mod.recommend_main_process_threads(cpu_count=128, num_workers=4) <= 4
    assert mod.recommend_main_process_threads(cpu_count=8, num_workers=0) <= 8
    assert mod.recommend_main_process_threads(cpu_count=8, num_workers=4) >= 1


def test_build_loader_kwargs_enables_persistent_workers_and_pin_memory_for_cuda() -> None:
    mod = _load_module()
    kwargs = mod.build_loader_kwargs(num_workers=4, use_cuda=True)
    assert kwargs["num_workers"] == 4
    assert kwargs["pin_memory"] is True
    assert kwargs["persistent_workers"] is True
    assert callable(kwargs["worker_init_fn"])
