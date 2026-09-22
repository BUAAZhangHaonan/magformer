from __future__ import annotations

from pathlib import Path

import pytest
import torch
import importlib.util


_UTILS_PATH = Path(__file__).resolve().parents[1] / "magformer" / "engine" / "utils.py"
_UTILS_SPEC = importlib.util.spec_from_file_location("magformer_engine_utils_test", _UTILS_PATH)
assert _UTILS_SPEC is not None and _UTILS_SPEC.loader is not None
engine_utils = importlib.util.module_from_spec(_UTILS_SPEC)
_UTILS_SPEC.loader.exec_module(engine_utils)

load_torch_checkpoint = engine_utils.load_torch_checkpoint
def test_load_torch_checkpoint_calls_weights_only_once(monkeypatch, tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.pth"
    calls: list[dict[str, object]] = []

    def fake_load(path, map_location=None, weights_only=None):
        calls.append({"path": str(path), "map_location": map_location, "weights_only": weights_only})
        return {"state_dict": {"weight": torch.tensor([1.0])}}

    monkeypatch.setattr(torch, "load", fake_load)

    checkpoint = load_torch_checkpoint(weights_path)

    assert checkpoint["state_dict"]["weight"].item() == pytest.approx(1.0)
    assert calls == [
        {
            "path": str(weights_path.resolve()),
            "map_location": "cpu",
            "weights_only": True,
        }
    ]


def test_load_torch_checkpoint_propagates_loader_errors(monkeypatch, tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.pth"

    def fake_load(path, map_location=None, weights_only=None):
        raise TypeError("weights_only unsupported")

    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(TypeError, match="weights_only unsupported"):
        load_torch_checkpoint(weights_path)
