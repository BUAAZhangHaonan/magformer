from __future__ import annotations

from types import SimpleNamespace

from baselines.detectron2.detectron2.engine.defaults import _ddp_kwargs_from_cfg


def test_ddp_kwargs_from_cfg_defaults_to_broadcast_buffers_only() -> None:
    cfg = SimpleNamespace()

    kwargs = _ddp_kwargs_from_cfg(cfg)

    assert kwargs == {"broadcast_buffers": False}


def test_ddp_kwargs_from_cfg_honors_find_unused_parameters_flag() -> None:
    cfg = SimpleNamespace(DDP=SimpleNamespace(FIND_UNUSED_PARAMETERS=True))

    kwargs = _ddp_kwargs_from_cfg(cfg)

    assert kwargs["broadcast_buffers"] is False
    assert kwargs["find_unused_parameters"] is True
