from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mgm_depth_mapper_supports_per_sample_norm_to_unit_interval() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mapper_mod = _load_module(
        "mgm_depth_utils",
        (
            repo_root
            / "baselines"
            / "MGM_Mask2Former"
            / "mask2former"
            / "data"
            / "depth_utils.py"
        ).resolve(),
    )

    depth = np.array(
        [[0.938, 0.940], [0.950, 0.962]],
        dtype=np.float32,
    )
    cfg = SimpleNamespace(
        INPUT=SimpleNamespace(
            DEPTH_SCALE=1.0,
            DEPTH_SHIFT=0.0,
            DEPTH_CLIP_MIN=0.9374,
            DEPTH_CLIP_MAX=0.9619,
            DEPTH_NORM="minmax",
            DEPTH_PER_SAMPLE_NORM=True,
        )
    )

    out = mapper_mod.normalize_and_augment_depth(depth, cfg)

    assert out.shape == (2, 2, 1)
    assert float(out.min()) == 0.0
    assert 0.99 <= float(out.max()) <= 1.0


def test_mgm_feature_helper_skips_depth_when_disabled() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    depth_path_mod = _load_module(
        "mgm_depth_path",
        (
            repo_root
            / "baselines"
            / "MGM_Mask2Former"
            / "mask2former"
            / "modeling"
            / "meta_arch"
            / "depth_path.py"
        ).resolve(),
    )

    class DummyBackbone(torch.nn.Module):
        def __init__(self, value: float):
            super().__init__()
            self.value = value
            self.calls = 0

        def forward(self, tensor):
            self.calls += 1
            return {"res3": torch.full((1, 1, 2, 2), self.value, dtype=torch.float32)}

    class DummyMgm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return {"res3": torch.full((1, 1, 2, 2), 9.0)}, {"res3": torch.ones(1, 1, 2, 2)}, {}

    rgb_backbone = DummyBackbone(3.0)
    depth_backbone = DummyBackbone(5.0)
    mgm = DummyMgm()

    fused_features, confidence_maps, losses = depth_path_mod.forward_fused_features(
        rgb_backbone=rgb_backbone,
        depth_backbone=depth_backbone,
        mgm=mgm,
        images_tensor=torch.zeros(1, 3, 2, 2),
        depths_tensor=torch.zeros(1, 1, 2, 2),
        depth_raw=torch.zeros(1, 1, 2, 2),
        rgb_image=torch.zeros(1, 3, 2, 2),
        depth_noise_mask=None,
        use_depth_path=False,
    )

    assert rgb_backbone.calls == 1
    assert depth_backbone.calls == 0
    assert mgm.calls == 0
    assert confidence_maps is None
    assert losses == {}
    assert torch.equal(fused_features["res3"], torch.full((1, 1, 2, 2), 3.0))
