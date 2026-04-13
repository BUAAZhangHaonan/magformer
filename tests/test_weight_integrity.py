from __future__ import annotations

import hashlib
import warnings
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
verify_sha256_sidecar = engine_utils.verify_sha256_sidecar


def _write_payload(path: Path, data: bytes = b"payload") -> str:
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_verify_sha256_sidecar_accepts_matching_hash(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.pth"
    digest = _write_payload(weights_path, b"good-weights")
    (tmp_path / "weights.pth.sha256").write_text(f"{digest}  weights.pth\n", encoding="utf-8")

    verify_sha256_sidecar(weights_path)


def test_verify_sha256_sidecar_rejects_mismatch(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.pth"
    _write_payload(weights_path, b"good-weights")
    (tmp_path / "weights.pth.sha256").write_text(
        f"{'0' * 64}  weights.pth\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_sha256_sidecar(weights_path)


def test_verify_sha256_sidecar_warns_once_when_missing(tmp_path: Path) -> None:
    engine_utils._MISSING_SHA256_WARNED_PATHS.clear()
    weights_path = tmp_path / "weights.pth"
    _write_payload(weights_path, b"good-weights")

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", UserWarning)
        verify_sha256_sidecar(weights_path)
        verify_sha256_sidecar(weights_path)

    assert len(records) == 1
    assert "Integrity verification is available but not configured" in str(records[0].message)


def test_load_torch_checkpoint_falls_back_with_warning(monkeypatch, tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.pth"
    weights_path.write_bytes(b"not-a-real-torch-checkpoint")

    calls: list[dict[str, object]] = []

    def fake_load(path, map_location=None, weights_only=None):
        calls.append({"path": str(path), "map_location": map_location, "weights_only": weights_only})
        if weights_only is True:
            raise TypeError("weights_only unsupported")
        return {"state_dict": {"weight": torch.tensor([1.0])}}

    monkeypatch.setattr(torch, "load", fake_load)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", UserWarning)
        checkpoint = load_torch_checkpoint(weights_path, verify_sha256=False)

    assert checkpoint["state_dict"]["weight"].item() == pytest.approx(1.0)
    assert calls[0]["weights_only"] is True
    assert calls[1]["weights_only"] is None
    assert any("Falling back to weights_only=False" in str(item.message) for item in records)
