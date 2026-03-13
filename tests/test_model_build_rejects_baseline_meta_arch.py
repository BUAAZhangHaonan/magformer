from __future__ import annotations

from types import SimpleNamespace

import pytest

from magformer.models.build import build_model


@pytest.mark.parametrize(
    "arch",
    [
        "ucn",
        "UCN",
        "msmformer",
        "MSMFormer",
        "uoa_is",
        "UOAIS",
        "uoa-is",
    ],
)
def test_build_model_rejects_external_baseline_meta_architectures(arch: str) -> None:
    config = SimpleNamespace(model=SimpleNamespace(meta_architecture=arch))
    with pytest.raises(ValueError, match="External baselines are not built through magformer.models.build"):
        build_model(config)  # type: ignore[arg-type]
