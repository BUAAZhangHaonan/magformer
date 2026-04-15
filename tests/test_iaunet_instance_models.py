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


def test_iaunet_matcher_and_losses_return_expected_shapes() -> None:
    mod = _load_module()
    model = mod.IAUNetInstanceModel(
        in_channels=3,
        base_channels=8,
        hidden_dim=32,
        num_queries=6,
        num_decoder_layers=2,
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

    assert {"loss_ce", "loss_mask", "loss_dice"} <= set(losses)
    for key in ("loss_ce", "loss_mask", "loss_dice"):
        assert losses[key].ndim == 0
        assert torch.isfinite(losses[key])


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

    outputs = {"pred_logits": pred_logits, "pred_masks": pred_masks}
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
    assert float(predictions[0]["scores"][0]) > 0.9
