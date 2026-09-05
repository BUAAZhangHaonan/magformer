from __future__ import annotations

import pytest

from magformer.config import load_config
from magformer.models import build_model
from magformer.models.magformer.arch import MagFormerArch


C0_CONFIG = "configs/next_stage/c0_seed42.yaml"


def test_nonzero_retired_small_object_sampling_knob_fails_before_model_init() -> None:
    config = load_config(C0_CONFIG)

    assert config.model.small_object_sample_threshold == 200
    with pytest.raises(
        ValueError,
        match=(
            r"model\.small_object_sample_threshold is a retired, unsupported "
            r"criterion sampling heuristic"
        ),
    ):
        build_model(config)


def test_explicit_zero_small_object_sampling_knob_allows_model_init(monkeypatch) -> None:
    config = load_config(
        C0_CONFIG,
        overrides={"model": {"small_object_sample_threshold": 0}},
    )
    sentinel = object()
    monkeypatch.setattr(
        MagFormerArch,
        "from_config",
        classmethod(lambda cls, cfg: sentinel),
    )

    assert config.model.small_object_sample_threshold == 0
    assert build_model(config) is sentinel
