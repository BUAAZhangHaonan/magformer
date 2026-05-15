from __future__ import annotations

import pytest
import torch

from magformer.models.ops.modules.ms_deform_attn import MSDeformAttn


def _small_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    shapes = torch.tensor([(4, 4), (2, 2)], dtype=torch.long)
    starts = torch.cat([shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]])
    total = int(shapes.prod(1).sum().item())
    query = torch.randn(1, total, 32, requires_grad=True)
    value = torch.randn(1, total, 32, requires_grad=True)
    reference_points = torch.rand(1, total, len(shapes), 2)
    return query, reference_points, value, shapes, starts


def test_explicit_pytorch_backend_is_differentiable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGFORMER_MS_DEFORM_ATTN_BACKEND", "pytorch")
    module = MSDeformAttn(d_model=32, n_levels=2, n_heads=4, n_points=2)
    query, reference_points, value, shapes, starts = _small_inputs()

    output = module(query, reference_points, value, shapes, starts)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == (1, 20, 32)
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def test_invalid_ms_deform_attn_backend_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGFORMER_MS_DEFORM_ATTN_BACKEND", "silent_fallback")
    module = MSDeformAttn(d_model=32, n_levels=2, n_heads=4, n_points=2)
    query, reference_points, value, shapes, starts = _small_inputs()

    with pytest.raises(ValueError, match="MAGFORMER_MS_DEFORM_ATTN_BACKEND"):
        module(query, reference_points, value, shapes, starts)
