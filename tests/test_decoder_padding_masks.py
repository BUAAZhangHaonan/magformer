from __future__ import annotations

from typing import List

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from magformer.models.common.layers.depth_position_encoding import DepthPosEncoding
from magformer.models.common.layers.position_encoding import PositionEmbeddingSine
from magformer.models.common.pixel_decoder_msdeformattn import MSDeformAttnPixelDecoder
from magformer.models.common.transformer.multiscale_decoder import (
    MultiScaleMaskedTransformerDecoder,
)
from magformer.models.common.transformer.dynamic_query import DynamicContentQueryModule
from magformer.models.magformer.arch import MagFormerArch


def _build_decoder(
    *,
    num_feature_levels: int = 1,
    use_checkpoint: bool = False,
    use_deformable_cross_attn: bool = False,
    mask_attn_topk_ratio: float | None = None,
) -> MultiScaleMaskedTransformerDecoder:
    return MultiScaleMaskedTransformerDecoder(
        num_queries=3,
        hidden_dim=8,
        nheads=2,
        dim_feedforward=16,
        num_layers=1,
        num_classes=1,
        mask_dim=8,
        dropout=0.0,
        num_feature_levels=num_feature_levels,
        mask_attn_topk_ratio=mask_attn_topk_ratio,
        mask_attn_topk_min=2,
        use_checkpoint=use_checkpoint,
        use_deformable_cross_attn=use_deformable_cross_attn,
        deformable_n_points=2,
    )


def test_position_embedding_is_padding_aware() -> None:
    pe = PositionEmbeddingSine(num_pos_feats=4, normalize=True)
    small = torch.zeros((1, 8, 2, 3))
    small_mask = torch.zeros((1, 2, 3), dtype=torch.bool)
    large = torch.zeros((1, 8, 4, 5))
    large_mask = torch.ones((1, 4, 5), dtype=torch.bool)
    large_mask[:, :2, :3] = False

    small_pos = pe(small, small_mask)
    large_pos = pe(large, large_mask)

    torch.testing.assert_close(small_pos, large_pos[:, :, :2, :3])
    assert torch.count_nonzero(large_pos.masked_select(large_mask.unsqueeze(1))) == 0


def test_depth_position_encoding_zeros_padding_after_projection() -> None:
    torch.manual_seed(0)
    dpe = DepthPosEncoding(hidden_dim=8)
    mask = torch.zeros((1, 4, 4), dtype=torch.bool)
    mask[:, :, 2:] = True
    depth_a = torch.rand((1, 1, 4, 4))
    depth_b = depth_a.clone()
    depth_b[:, :, :, 2:] = 1e6

    output_a = dpe(depth_a, mask)
    output_b = dpe(depth_b, mask)

    assert torch.count_nonzero(output_a.masked_select(mask.unsqueeze(1))) == 0
    assert torch.count_nonzero(output_b.masked_select(mask.unsqueeze(1))) == 0
    torch.testing.assert_close(
        output_a.masked_select((~mask).unsqueeze(1)),
        output_b.masked_select((~mask).unsqueeze(1)),
    )


class _PixelEncoderSpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.masks: List[torch.Tensor] = []
        self.positions: List[torch.Tensor] = []

    def forward(self, srcs, pos_embeds, masks):
        self.masks = [mask.detach().clone() for mask in masks]
        self.positions = [pos.detach().clone() for pos in pos_embeds]
        flattened = [src.flatten(2).transpose(1, 2) for src in srcs]
        memory = torch.cat(flattened, dim=1)
        spatial_shapes = torch.tensor(
            [src.shape[-2:] for src in srcs],
            dtype=torch.long,
            device=memory.device,
        )
        level_start_index = torch.cat(
            [
                torch.zeros((1,), dtype=torch.long, device=memory.device),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            ]
        )
        return memory, spatial_shapes, level_start_index


def test_pixel_decoder_returns_resized_masks_in_feature_order() -> None:
    decoder = MSDeformAttnPixelDecoder(
        in_features=["res2", "res3"],
        in_channels={"res2": 8, "res3": 8},
        transformer_in_features=["res2", "res3"],
        hidden_dim=8,
        mask_dim=8,
        transformer_dropout=0.0,
        transformer_nheads=2,
        transformer_dim_feedforward=16,
        transformer_enc_layers=1,
        maskformer_num_feature_levels=2,
    )
    spy = _PixelEncoderSpy()
    decoder.transformer = spy
    features = {
        "res2": torch.randn((1, 8, 4, 6)),
        "res3": torch.randn((1, 8, 2, 3)),
    }
    padding = torch.ones((1, 8, 12), dtype=torch.bool)
    padding[:, :6, :8] = False

    result = decoder(features, padding_mask=padding)
    masks = result["multi_scale_padding_masks"]

    assert [tuple(mask.shape[-2:]) for mask in masks] == [(2, 3), (4, 6)]
    expected = [
        F.interpolate(padding[:, None].float(), size=size, mode="nearest")
        .squeeze(1)
        .bool()
        for size in [(2, 3), (4, 6)]
    ]
    for actual, wanted, encoder_mask, position in zip(
        masks, expected, spy.masks, spy.positions
    ):
        assert actual.dtype == torch.bool
        assert torch.equal(actual, wanted)
        assert torch.equal(actual, encoder_mask)
        assert torch.count_nonzero(position.masked_select(actual.unsqueeze(1))) == 0


def test_pixel_decoder_rejects_all_padding_level() -> None:
    feature = torch.zeros((1, 8, 2, 2))
    with pytest.raises(ValueError, match="all-padding feature level"):
        MSDeformAttnPixelDecoder._resize_padding_mask(
            torch.ones((1, 4, 4), dtype=torch.bool),
            feature,
            level_name="res5",
        )


def test_dense_all_masked_recovery_reopens_only_valid_keys() -> None:
    decoder = _build_decoder()
    semantic = torch.ones((2, 3, 4), dtype=torch.bool)
    padding = torch.tensor([[False, True, False, True]])

    combined = decoder._combine_semantic_and_padding_mask(semantic, padding)

    expected = padding[:, None, None].expand(1, 2, 3, 4).reshape(2, 3, 4)
    assert torch.equal(combined, expected)
    assert not combined.all(dim=-1).any()


def test_dense_decoder_is_invariant_to_padding_values() -> None:
    torch.manual_seed(1)
    decoder = _build_decoder().eval()
    feature_a = torch.randn((1, 8, 2, 3))
    feature_b = feature_a.clone()
    padding = torch.tensor([[[False, False, True], [False, False, True]]])
    feature_b.masked_fill_(padding.unsqueeze(1), 1e5)
    position = PositionEmbeddingSine(4, normalize=True)(feature_a, padding)
    mask_features = torch.randn((1, 8, 4, 6))

    kwargs = {
        "memory": feature_a,
        "mask_features": mask_features,
        "multi_scale_pos": [position],
        "multi_scale_padding_masks": [padding],
    }
    output_a = decoder(multi_scale_features=[feature_a], **kwargs)
    output_b = decoder(multi_scale_features=[feature_b], **kwargs)

    for key in ("pred_logits", "pred_masks", "pred_boxes"):
        torch.testing.assert_close(output_a[key], output_b[key])


def test_dynamic_queries_use_valid_pixel_pooling() -> None:
    torch.manual_seed(11)
    module = DynamicContentQueryModule(
        num_queries=3,
        hidden_dim=8,
        num_feature_levels=1,
        num_patterns=4,
        context_dim=8,
    ).eval()
    feature_a = torch.randn((1, 8, 2, 3))
    feature_b = feature_a.clone()
    padding = torch.tensor([[[False, False, True], [False, False, True]]])
    feature_b.masked_fill_(padding.unsqueeze(1), -1e5)
    static_queries = torch.randn((3, 8))

    output_a = module([feature_a], static_queries, [padding])
    output_b = module([feature_b], static_queries, [padding])

    torch.testing.assert_close(output_a, output_b)


def test_topk_attention_never_ranks_padding_as_valid() -> None:
    torch.manual_seed(2)
    decoder = _build_decoder(mask_attn_topk_ratio=1.0)
    output = torch.randn((3, 1, 8))
    mask_features = torch.randn((1, 8, 2, 3))
    padding = torch.tensor([[[False, False, True], [False, False, True]]])

    _, _, attn_mask, _ = decoder.forward_prediction_heads(
        output,
        mask_features,
        (2, 3),
        padding,
    )

    attn_mask = attn_mask.reshape(1, 2, 3, 6)
    padding_flat = padding.flatten(1)
    assert attn_mask[..., padding_flat[0]].all()
    assert (~attn_mask[..., ~padding_flat[0]]).any()


class _DeformableAttentionSpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.received_masks: List[torch.Tensor] = []

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        spatial_shapes,
        level_start_index,
        input_padding_mask,
    ):
        self.received_masks.append(input_padding_mask.detach().clone())
        valid = (~input_padding_mask).unsqueeze(-1).to(input_flatten.dtype)
        pooled = (input_flatten * valid).sum(dim=1) / valid.sum(dim=1)
        return query + self.scale * pooled.unsqueeze(0)


def test_deformable_checkpoint_path_receives_padding_mask_and_backpropagates() -> None:
    torch.manual_seed(3)
    decoder = _build_decoder(
        use_checkpoint=True,
        use_deformable_cross_attn=True,
    ).train()
    spy = _DeformableAttentionSpy()
    decoder.transformer_cross_attention_layers[0] = spy
    feature = torch.randn((1, 8, 2, 3), requires_grad=True)
    padding = torch.tensor([[[False, False, True], [False, False, True]]])
    mask_features = torch.randn((1, 8, 4, 6), requires_grad=True)

    outputs = decoder(
        memory=feature,
        mask_features=mask_features,
        multi_scale_features=[feature],
        multi_scale_pos=[torch.zeros_like(feature)],
        multi_scale_padding_masks=[padding],
    )
    loss = outputs["pred_logits"].sum() + outputs["pred_masks"].sum()
    loss.backward()

    assert spy.received_masks
    assert all(torch.equal(mask, padding.flatten(1)) for mask in spy.received_masks)
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    assert spy.scale.grad is not None and torch.isfinite(spy.scale.grad)


class _RgbBackbone(nn.Module):
    def forward(self, images: torch.Tensor):
        batch_size = images.shape[0]
        return {
            "res2": torch.zeros((batch_size, 8, 8, 8), device=images.device),
            "res3": torch.zeros((batch_size, 8, 4, 4), device=images.device),
            "res4": torch.zeros((batch_size, 8, 2, 2), device=images.device),
            "res5": torch.zeros((batch_size, 8, 1, 1), device=images.device),
        }


class _UnusedModule(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("disabled module must not be called")


class _PixelDecoderForArch(nn.Module):
    def forward(
        self,
        features,
        confidence_maps=None,
        depth_modulation_maps=None,
        depth_raw=None,
        padding_mask=None,
    ):
        feature = features["res2"]
        level_mask = F.interpolate(
            padding_mask[:, None].float(),
            size=feature.shape[-2:],
            mode="nearest",
        ).squeeze(1).bool()
        return {
            "memory": feature,
            "mask_features": feature,
            "multi_scale_features": [feature],
            "multi_scale_pos": [torch.zeros_like(feature)],
            "multi_scale_padding_masks": [level_mask],
        }


class _DecoderForArch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received: List[List[torch.Tensor]] = []

    def forward(self, **kwargs):
        masks = kwargs["multi_scale_padding_masks"]
        self.received.append([mask.detach().clone() for mask in masks])
        batch_size = kwargs["memory"].shape[0]
        device = kwargs["memory"].device
        return {
            "pred_logits": torch.zeros((batch_size, 2, 2), device=device),
            "pred_masks": torch.zeros((batch_size, 2, 8, 8), device=device),
            "pred_boxes": torch.zeros((batch_size, 2, 4), device=device),
            "aux_outputs": [],
        }


class _CriterionForArch(nn.Module):
    def forward(self, outputs, targets):
        return {"total_loss": outputs["pred_logits"].sum()}


def test_arch_passes_padding_masks_through_all_decoder_calls() -> None:
    transformer_decoder = _DecoderForArch()
    arch = MagFormerArch(
        rgb_backbone=_RgbBackbone(),
        depth_backbone=_UnusedModule(),
        fusion_module=_UnusedModule(),
        pixel_decoder=_PixelDecoderForArch(),
        transformer_decoder=transformer_decoder,
        num_classes=1,
        num_queries=2,
        hidden_dim=8,
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )
    arch.modality_fusion_enabled = False
    arch.depth_backbone_enabled = False
    arch.criterion = _CriterionForArch()
    images = torch.zeros((1, 3, 32, 32))
    depths = torch.zeros((1, 1, 32, 32))
    padding = torch.ones((1, 32, 32), dtype=torch.bool)
    padding[:, :24, :20] = False

    arch.train()
    arch(
        images,
        depths,
        targets=[
            {
                "labels": torch.tensor([0]),
                "masks": torch.zeros((1, 32, 32)),
            }
        ],
        padding_masks=padding,
    )
    arch.eval()
    arch(images, depths, padding_masks=padding, return_features=True)
    arch.forward_inference_decoder_outputs(images, depths, padding_masks=padding)
    arch.collect_preflight_diagnostics(images, depths, padding_masks=padding)

    assert len(transformer_decoder.received) == 4
    expected = F.interpolate(
        padding[:, None].float(), size=(8, 8), mode="nearest"
    ).squeeze(1).bool()
    for masks in transformer_decoder.received:
        assert len(masks) == 1
        assert torch.equal(masks[0], expected)
