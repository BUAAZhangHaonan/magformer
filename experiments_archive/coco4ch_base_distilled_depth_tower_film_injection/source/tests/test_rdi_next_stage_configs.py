from copy import deepcopy
from pathlib import Path

import pytest
import torch

from magformer.config import load_config
from magformer.models import build_model


CONFIG_DIR = Path("configs/next_stage")
CONFIGS = (
    CONFIG_DIR / "rdi_lite_300k_seed42.yaml",
    CONFIG_DIR / "rdi_lite_300k_seed43.yaml",
    CONFIG_DIR / "rdi_mbv3l_300k_seed42.yaml",
    CONFIG_DIR / "rdi_mbv3l_300k_seed43.yaml",
)


def _protocol_signature(config):
    data = deepcopy(config.model_dump())
    data.pop("name", None)
    runtime = data["runtime"]
    runtime.pop("seed")
    runtime.pop("output_dir")
    magformer = data["model"]["magformer"]
    magformer["residual_depth_fusion"].pop("encoder")
    depth_backbone = magformer["depth_backbone"]
    depth_backbone.pop("name")
    depth_backbone.pop("weights")
    return data


def test_rdi_configs_share_one_training_protocol():
    configs = [load_config(str(path)) for path in CONFIGS]
    reference = _protocol_signature(configs[0])

    assert all(_protocol_signature(config) == reference for config in configs[1:])
    for config in configs:
        assert config.data.dataset_root.endswith("20260318_1K_32254")
        assert config.solver.max_iter == 300000
        assert config.runtime.eval_period == 37500
        assert config.runtime.checkpoint_period == 37500
        assert config.runtime.num_workers == 2
        assert config.runtime.cuda_memory_fraction == pytest.approx(0.85)
        assert config.model.magformer.modality_fusion.enabled is False
        assert config.model.magformer.dpe.enabled is False
        assert config.model.magformer.rgb_backbone.weights is None
        depth_backbone = config.model.magformer.depth_backbone
        if config.model.magformer.residual_depth_fusion.encoder == "light_depth_pyramid":
            assert depth_backbone.name == "light_depth_pyramid"
            assert depth_backbone.weights is None
        else:
            assert depth_backbone.name == "mobilenetv3_large"
            assert depth_backbone.weights.endswith(
                "mobilenetv3_large_100_depth.pth"
            )


@pytest.mark.parametrize(
    "path,encoder",
    (
        (CONFIGS[0], "light_depth_pyramid"),
        (CONFIGS[1], "light_depth_pyramid"),
        (CONFIGS[2], "mobilenetv3_large"),
        (CONFIGS[3], "mobilenetv3_large"),
    ),
)
def test_rdi_config_builds_only_the_unified_depth_path(path, encoder):
    config = load_config(str(path))
    model = build_model(config)
    parameter_names = tuple(name for name, _ in model.named_parameters())

    assert config.model.magformer.residual_depth_fusion.encoder == encoder
    assert not any(name.startswith("fusion.") for name in parameter_names)
    assert not any(name.startswith("depth_backbone.") for name in parameter_names)
    assert not any("depth_pe" in name for name in parameter_names)
    assert not any("confidence" in name for name in parameter_names)
    assert not any("dccg" in name.lower() for name in parameter_names)
    assert any(
        name.startswith("rgb_backbone.depth_encoder.")
        for name in parameter_names
    )
    forbidden_loss_tokens = ("dccg", "confidence", "entropy", "dpe")
    assert not any(
        token in loss_name.lower()
        for loss_name in model.criterion.weight_dict
        for token in forbidden_loss_tokens
    )
    expected_total = (
        44_752_429 if encoder == "light_depth_pyramid" else 48_291_381
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_total
    for projection in model.rgb_backbone.depth_projections.values():
        assert torch.count_nonzero(projection.weight).item() == 0
        assert torch.count_nonzero(projection.bias).item() == 0
