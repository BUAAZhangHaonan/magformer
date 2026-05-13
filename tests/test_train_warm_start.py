from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tools.train import (
    _extract_model_state_dict,
    _strip_module_prefix_if_needed,
    load_finetune_weights,
    resolve_checkpoint_init_mode,
)
from magformer.engine.utils import load_checkpoint


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)


PRETRAINED_512_CHECKPOINT = Path(
    "output/pretrained/model_final_3c8ec9_1class_v2_ecc20260318_1k_1566_512.pth"
)
TEACHER_8499_CHECKPOINT = Path(
    "output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth"
)


def _load_checkpoint_state_for_smoke(path: Path):
    if not path.exists():
        pytest.skip(f"checkpoint not available for local smoke: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = _strip_module_prefix_if_needed(_extract_model_state_dict(checkpoint))
    tensor_keys = [k for k, v in state.items() if torch.is_tensor(v)]
    return checkpoint, state, tensor_keys


def test_resolve_checkpoint_init_mode_prioritizes_resume():
    assert resolve_checkpoint_init_mode("resume.pth", "finetune.pth") == "resume"
    assert resolve_checkpoint_init_mode("resume.pth", None) == "resume"
    assert resolve_checkpoint_init_mode(None, "finetune.pth") == "finetune"
    assert resolve_checkpoint_init_mode(None, None) == "none"


def test_load_finetune_weights_loads_model_only_and_reports_keys(tmp_path):
    model = TinyModel()
    checkpoint_path = Path(tmp_path) / "warm_start.pth"

    model_state = model.state_dict()
    model_state["linear.weight"] = torch.ones_like(model_state["linear.weight"])
    model_state["unexpected.weight"] = torch.ones(1)
    torch.save({"model_state_dict": model_state, "optimizer_state_dict": {"ignored": True}}, checkpoint_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer_state_before = optimizer.state_dict()

    load_info = load_finetune_weights(model, str(checkpoint_path), strict=False)

    assert torch.allclose(model.linear.weight, torch.ones_like(model.linear.weight))
    assert "unexpected.weight" in load_info["unexpected_keys"]
    assert optimizer.state_dict() == optimizer_state_before


def test_extract_model_state_dict_supports_top_level_model_checkpoint():
    model_state = {
        "linear.weight": torch.ones(2, 4),
        "linear.bias": torch.zeros(2),
    }
    checkpoint = {"model": model_state, "__author__": "detectron2", "source": "unit-test"}

    assert _extract_model_state_dict(checkpoint) is model_state


def test_extract_model_state_dict_rejects_checkpoint_without_model_weights():
    checkpoint = {"optimizer_state_dict": {"ignored": True}, "epoch": 3}

    with pytest.raises(ValueError, match="model_state_dict|state_dict|model"):
        _extract_model_state_dict(checkpoint)


def test_load_finetune_weights_loads_top_level_model_checkpoint(tmp_path):
    model = TinyModel()
    checkpoint_path = Path(tmp_path) / "detectron_model_checkpoint.pth"

    model_state = model.state_dict()
    model_state["linear.weight"] = torch.full_like(model_state["linear.weight"], 7.0)
    model_state["linear.bias"] = torch.full_like(model_state["linear.bias"], 9.0)
    torch.save({"model": model_state, "__author__": "detectron2", "source": "unit-test"}, checkpoint_path)

    load_info = load_finetune_weights(model, str(checkpoint_path), strict=False)

    assert torch.allclose(model.linear.weight, torch.full_like(model.linear.weight, 7.0))
    assert torch.allclose(model.linear.bias, torch.full_like(model.linear.bias, 9.0))
    assert "model" not in load_info["unexpected_keys"]
    assert "__author__" not in load_info["unexpected_keys"]
    assert "source" not in load_info["unexpected_keys"]


def test_load_finetune_weights_rejects_low_match_ratio(tmp_path):
    model = TinyModel()
    checkpoint_path = Path(tmp_path) / "unrelated_checkpoint.pth"

    torch.save({"state_dict": {"other.weight": torch.ones(1)}}, checkpoint_path)

    with pytest.raises(RuntimeError, match="matched.*expected"):
        load_finetune_weights(model, str(checkpoint_path), strict=False)


def test_pretrained_512_checkpoint_unpacks_top_level_model_on_cpu():
    checkpoint, state, tensor_keys = _load_checkpoint_state_for_smoke(PRETRAINED_512_CHECKPOINT)

    assert list(checkpoint.keys())[:3] == ["model", "__author__", "source"]
    assert len(tensor_keys) > 100
    assert "model" not in state
    assert any(key.startswith("backbone.") for key in tensor_keys)


def test_teacher_8499_checkpoint_unpacks_model_state_dict_on_cpu():
    checkpoint, state, tensor_keys = _load_checkpoint_state_for_smoke(TEACHER_8499_CHECKPOINT)

    assert "model_state_dict" in checkpoint
    assert len(tensor_keys) > 100
    assert "model_state_dict" not in state
    assert any(key.startswith("rgb_backbone.") for key in tensor_keys)


def test_load_checkpoint_strips_ddp_module_prefix(tmp_path):
    model = TinyModel()
    checkpoint_path = Path(tmp_path) / "ddp_checkpoint.pth"

    prefixed_state = {
        "module.linear.weight": torch.full_like(model.linear.weight, 3.0),
        "module.linear.bias": torch.full_like(model.linear.bias, 5.0),
    }
    torch.save({"model_state_dict": prefixed_state}, checkpoint_path)

    load_checkpoint(str(checkpoint_path), model, strict=False)

    assert torch.allclose(model.linear.weight, torch.full_like(model.linear.weight, 3.0))
    assert torch.allclose(model.linear.bias, torch.full_like(model.linear.bias, 5.0))
