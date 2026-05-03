from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "baselines" / "iaunet_instance_models.py"
    spec = importlib.util.spec_from_file_location("iaunet_instance_models", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_targets() -> list[dict[str, torch.Tensor]]:
    masks_a = torch.zeros((2, 32, 32), dtype=torch.float32)
    masks_a[0, 3:13, 4:14] = 1.0
    masks_a[1, 18:28, 19:29] = 1.0

    masks_b = torch.zeros((1, 32, 32), dtype=torch.float32)
    masks_b[0, 10:24, 8:20] = 1.0

    return [
        {"labels": torch.ones((2,), dtype=torch.int64), "masks": masks_a},
        {"labels": torch.ones((1,), dtype=torch.int64), "masks": masks_b},
    ]


def test_iaunet_default_architecture_matches_paper_contract() -> None:
    mod = _load_module()
    model = mod.IAUNetInstanceModel()
    assert model.hidden_dim == 256
    assert model.query_decoder.query_embed.num_embeddings == 100
    assert model.query_decoder.blocks_per_stage == 3
    assert len(model.query_decoder.stage_blocks) == 4
    assert len(model.pixel_decoder.skip_projections) == 4
    assert any(isinstance(module, mod.CoordConv) for module in model.modules())
    assert any(isinstance(module, mod.SqueezeExcitationBlock) for module in model.modules())
    assert model.paper_faithful is True


def test_iaunet_bce_matcher_cost_matches_expanded_reference() -> None:
    mod = _load_module()
    pred_logits = torch.randn(5, 7, 9, dtype=torch.float32)
    target_masks = (torch.rand(4, 7, 9) > 0.35).float()

    pred = pred_logits.flatten(1)
    target = target_masks.flatten(1)
    pred_prob = pred.sigmoid()[:, None, :].expand(-1, target.shape[0], -1)
    target_expanded = target[None, :, :].expand(pred.shape[0], -1, -1)
    expected = torch.nn.functional.binary_cross_entropy(pred_prob, target_expanded, reduction="none").mean(dim=-1)

    actual = mod._batch_sigmoid_bce_cost(pred_logits, target_masks)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_iaunet_matcher_handles_more_targets_than_queries() -> None:
    mod = _load_module()
    outputs = {
        "pred_logits": torch.randn(1, 2, 2),
        "pred_masks": torch.randn(1, 2, 8, 8),
    }
    target_masks = torch.zeros((4, 8, 8), dtype=torch.float32)
    target_masks[0, 1:3, 1:3] = 1.0
    target_masks[1, 3:5, 3:5] = 1.0
    target_masks[2, 5:7, 5:7] = 1.0
    target_masks[3, 2:6, 1:4] = 1.0

    indices = mod.IAUNetHungarianMatcher()(outputs, [{"labels": torch.ones((4,), dtype=torch.int64), "masks": target_masks}])

    assert len(indices) == 1
    assert indices[0][0].numel() == 2
    assert indices[0][1].numel() == 2


def test_iaunet_matcher_and_losses_return_expected_shapes() -> None:
    mod = _load_module()
    model = mod.IAUNetInstanceModel(
        in_channels=3,
        base_channels=8,
        hidden_dim=32,
        num_queries=6,
        num_decoder_layers=4,
        transformer_blocks_per_stage=1,
        num_heads=4,
    )
    images = torch.randn(2, 3, 32, 32)
    outputs = model(images)
    targets = _make_targets()

    matcher = mod.IAUNetHungarianMatcher()
    indices = matcher(outputs, targets)
    assert len(indices) == 2
    assert indices[0][0].shape == indices[0][1].shape
    assert indices[1][0].shape == indices[1][1].shape
    assert indices[0][0].numel() == 2
    assert indices[1][0].numel() == 1

    criterion = mod.IAUNetCriterion(matcher=matcher)
    losses = criterion(outputs, targets)

    assert {"loss_ce", "loss_mask", "loss_dice", "loss_maskness"} <= set(losses)
    for key in ("loss_ce", "loss_mask", "loss_dice", "loss_maskness"):
        assert losses[key].ndim == 0
        assert torch.isfinite(losses[key])

    assert len(outputs["aux_outputs"]) == 3


def test_iaunet_default_deep_supervision_after_each_transformer_block() -> None:
    mod = _load_module()
    model = mod.IAUNetInstanceModel(hidden_dim=32, num_queries=4, transformer_blocks_per_stage=3, num_heads=4)
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 64, 64))

    assert "pred_maskness" in outputs
    assert len(outputs["aux_outputs"]) == 11
    assert all("pred_maskness" in aux for aux in outputs["aux_outputs"])


def test_iaunet_inference_filters_low_confidence_and_empty_masks() -> None:
    mod = _load_module()
    pred_logits = torch.tensor(
        [
            [
                [0.0, 5.0],
                [5.0, 0.0],
                [0.0, 5.0],
            ]
        ],
        dtype=torch.float32,
    )
    pred_masks = torch.full((1, 3, 16, 16), -8.0, dtype=torch.float32)
    pred_masks[0, 0, 2:10, 3:12] = 8.0
    pred_maskness = torch.tensor([[[4.0], [4.0], [-4.0]]], dtype=torch.float32)

    outputs = {"pred_logits": pred_logits, "pred_masks": pred_masks, "pred_maskness": pred_maskness}
    predictions = mod.iaunet_inference(
        outputs,
        original_sizes=[(16, 16)],
        score_threshold=0.5,
        mask_threshold=0.5,
        min_area=8,
    )

    assert len(predictions) == 1
    assert predictions[0]["category_ids"].tolist() == [0]
    assert predictions[0]["masks"].shape == (1, 16, 16)
    assert predictions[0]["scores"].shape == (1,)
    assert 0.85 < float(predictions[0]["scores"][0]) < 0.99


def test_iaunet_maskness_rescoring_changes_confidence() -> None:
    mod = _load_module()
    pred_logits = torch.tensor([[[0.0, 5.0], [0.0, 5.0]]], dtype=torch.float32)
    pred_masks = torch.full((1, 2, 8, 8), 8.0, dtype=torch.float32)
    outputs = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks,
        "pred_maskness": torch.tensor([[[4.0], [-4.0]]], dtype=torch.float32),
    }

    predictions = mod.iaunet_inference(
        outputs,
        original_sizes=[(8, 8)],
        score_threshold=0.01,
        mask_threshold=0.5,
        min_area=1,
    )

    scores = predictions[0]["scores"].tolist()
    assert len(scores) == 2
    assert scores[0] > scores[1]
