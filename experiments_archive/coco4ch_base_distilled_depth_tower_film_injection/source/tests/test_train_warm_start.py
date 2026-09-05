from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn

from tools.train import (
    load_finetune_weights,
    resolve_checkpoint_init_mode,
)
from magformer.engine.utils import load_checkpoint


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)


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
