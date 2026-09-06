from pathlib import Path

import numpy as np
import torch

from magformer.data.transforms import DepthNormalize, ToTensor
from magformer.models.common.backbones.d2_swin import D2SwinBackbone
from magformer.models.common.backbones.mobilenet import MobileNetV3Depth
from magformer.models.common.backbones.residual_depth_swin import (
    LightDepthPyramid,
    ResidualDepthSwinBackbone,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MBV3_WEIGHTS = (
    REPO_ROOT / "pretrained_weights" / "mobilenetv3_large_100_depth.pth"
)


def _swin_kwargs():
    return {
        "embed_dim": 96,
        "depths": [1, 1, 1, 1],
        "num_heads": [3, 6, 12, 24],
        "window_size": 7,
        "drop_path_rate": 0.0,
        "out_features": ["res2", "res3", "res4", "res5"],
        "pretrained": False,
        "weights_path": None,
        "img_size": 128,
        "use_checkpoint": False,
    }


def _make_lite_rdi():
    return ResidualDepthSwinBackbone(
        depth_encoder=LightDepthPyramid(),
        depth_channels=[24, 48, 96, 192],
        **_swin_kwargs(),
    )


def _nonzero_finite_grad(parameters):
    grads = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    return bool(grads) and all(torch.isfinite(grad).all() for grad in grads) and any(
        torch.count_nonzero(grad).item() > 0 for grad in grads
    )


def test_depth_valid_mask_is_captured_before_normalized_zero():
    transform = DepthNormalize(
        scale=1.0,
        clip_min=0.3,
        clip_max=0.7,
        norm="minmax",
        per_sample_norm=False,
    )
    result = transform(
        {
            "depth": np.array(
                [[0.3, 0.5, 0.0], [np.nan, 0.7, np.inf]],
                dtype=np.float32,
            )
        }
    )
    assert result["depth"][0, 0] == 0.0
    assert result["depth_valid_mask"][0, 0]
    assert not result["depth_valid_mask"][0, 2]
    assert not result["depth_valid_mask"][1, 0]
    assert not result["depth_valid_mask"][1, 2]

    tensor_result = ToTensor()(result)
    assert tensor_result["depth_valid_mask"].shape == (1, 2, 3)
    assert tensor_result["depth_valid_mask"].dtype is torch.bool


def test_lite_rdi_identity_shapes_odd_size_and_gradients():
    torch.manual_seed(7)
    rdi = _make_lite_rdi().eval()
    rgb_only = D2SwinBackbone(**_swin_kwargs()).eval()
    rgb_only.model.load_state_dict(rdi.model.state_dict(), strict=True)

    rgb = torch.randn(1, 3, 65, 97)
    depth = torch.rand(1, 1, 65, 97)
    valid = torch.rand(1, 1, 65, 97) > 0.2
    padded_rgb, _, _ = rdi._pad_to_stride(rgb, depth, valid)

    with torch.no_grad():
        expected = rgb_only(padded_rgb)
        actual = rdi(rgb, depth, valid)

    expected_shapes = {
        "res2": (1, 96, 24, 32),
        "res3": (1, 192, 12, 16),
        "res4": (1, 384, 6, 8),
        "res5": (1, 768, 3, 4),
    }
    assert {key: tuple(value.shape) for key, value in actual.items()} == expected_shapes
    for key in expected:
        assert torch.equal(actual[key], expected[key])

    rdi.train()
    output = rdi(rgb, depth, valid)
    sum(value.square().mean() for value in output.values()).backward()
    assert _nonzero_finite_grad(rdi.depth_projections.parameters())
    assert not _nonzero_finite_grad(rdi.depth_encoder.parameters())

    optimizer = torch.optim.SGD(rdi.parameters(), lr=1.0e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = rdi(rgb, depth, valid)
    sum(value.square().mean() for value in output.values()).backward()
    assert _nonzero_finite_grad(rdi.depth_encoder.parameters())


def test_rdi_adapter_normalizes_and_stays_float32():
    torch.manual_seed(13)
    rdi = _make_lite_rdi().eval()
    projection_input = {}

    def capture_projection_input(_module, inputs):
        projection_input["value"] = inputs[0].detach()

    hook = rdi.depth_projections["res2"].register_forward_pre_hook(
        capture_projection_input
    )
    try:
        x = torch.randn(2, 64 * 64, 96, dtype=torch.float16)
        depth_feature = torch.randn(2, 24, 64, 64, dtype=torch.float16) * 9.0 + 4.0
        valid = torch.ones(2, 1, 256, 256, dtype=torch.bool)
        output = rdi._inject_stage(
            "res2",
            x,
            depth_feature,
            valid,
            (64, 64),
        )
    finally:
        hook.remove()

    normalized = projection_input["value"]
    rms = normalized.square().mean(dim=(1, 2, 3)).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=2.0e-5, rtol=0.0)
    assert normalized.dtype is torch.float32
    assert output.dtype is torch.float32
    assert torch.equal(output, x.float())


def test_projection_gradient_is_finite_and_nonzero():
    torch.manual_seed(17)
    rdi = _make_lite_rdi().train()
    x = torch.randn(1, 32 * 32, 96, requires_grad=True)
    depth_feature = torch.randn(1, 24, 32, 32, requires_grad=True)
    valid = torch.ones(1, 1, 128, 128, dtype=torch.bool)
    output = rdi._inject_stage(
        "res2",
        x,
        depth_feature,
        valid,
        (32, 32),
    )
    output.square().mean().backward()
    assert _nonzero_finite_grad(rdi.depth_projections["res2"].parameters())


def test_invalid_mask_zeroes_adapter_output():
    torch.manual_seed(19)
    rdi = _make_lite_rdi().eval()
    rdi.zero_grad(set_to_none=True)
    with torch.no_grad():
        rdi.depth_projections["res2"].weight.normal_(mean=0.0, std=0.1)
        rdi.depth_projections["res2"].bias.fill_(0.25)
    x = torch.randn(1, 32 * 32, 96)
    depth_feature = torch.randn(1, 24, 32, 32)
    valid = torch.zeros(1, 1, 128, 128, dtype=torch.bool)
    output = rdi._inject_stage(
        "res2",
        x,
        depth_feature,
        valid,
        (32, 32),
    )
    assert torch.equal(output, x.float())


def test_all_invalid_depth_has_strict_zero_injection():
    torch.manual_seed(11)
    rdi = _make_lite_rdi().eval()
    rgb_only = D2SwinBackbone(**_swin_kwargs()).eval()
    rgb_only.model.load_state_dict(rdi.model.state_dict(), strict=True)
    with torch.no_grad():
        for projection in rdi.depth_projections.values():
            projection.weight.normal_(mean=0.0, std=0.1)
            projection.bias.fill_(0.25)

    rgb = torch.randn(1, 3, 63, 95)
    depth = torch.rand(1, 1, 63, 95)
    valid = torch.zeros_like(depth, dtype=torch.bool)
    padded_rgb, _, _ = rdi._pad_to_stride(rgb, depth, valid)
    with torch.no_grad():
        expected = rgb_only(padded_rgb)
        actual = rdi(rgb, depth, valid)
    for key in expected:
        assert torch.equal(actual[key], expected[key])


def test_mbv3_large_checkpoint_is_an_exact_match_and_rdi_shapes():
    assert MBV3_WEIGHTS.is_file(), MBV3_WEIGHTS
    depth_encoder = MobileNetV3Depth(
        variant="large",
        out_features=["res2", "res3", "res4", "res5"],
        pretrained=False,
        weights_path=str(MBV3_WEIGHTS),
    )
    report = depth_encoder.pretrained_load_report
    assert report is not None
    assert report["matched"] == report["expected"] == 308
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert report["shape_mismatch_keys"] == []

    rdi = ResidualDepthSwinBackbone(
        depth_encoder=depth_encoder,
        depth_channels=[24, 40, 112, 960],
        **_swin_kwargs(),
    ).eval()
    rgb = torch.randn(1, 3, 65, 97)
    depth = torch.rand(1, 1, 65, 97)
    valid = torch.ones_like(depth, dtype=torch.bool)
    with torch.no_grad():
        outputs = rdi(rgb, depth, valid)
    assert [outputs[key].shape[1] for key in ("res2", "res3", "res4", "res5")] == [
        96,
        192,
        384,
        768,
    ]
