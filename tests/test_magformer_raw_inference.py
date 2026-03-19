from __future__ import annotations

import numpy as np
import torch

from magformer.models.magformer.arch import MagFormerArch


def test_inference_raw_keeps_tensor_predictions() -> None:
    outputs = {
        "pred_logits": torch.tensor([[[8.0, -2.0], [7.0, -1.0]]], dtype=torch.float32),
        "pred_masks": torch.randn(1, 2, 8, 8, dtype=torch.float32),
    }

    raw = MagFormerArch._inference_raw(outputs, (1, 3, 8, 8))

    pred = raw["predictions"][0]
    assert torch.is_tensor(pred["scores"])
    assert torch.is_tensor(pred["category_ids"])
    assert torch.is_tensor(pred["masks"])
    assert "pred_logits" not in raw
    assert "pred_masks" not in raw


def test_inference_raw_can_keep_raw_tensors() -> None:
    outputs = {
        "pred_logits": torch.tensor([[[8.0, -2.0], [7.0, -1.0]]], dtype=torch.float32),
        "pred_masks": torch.randn(1, 2, 8, 8, dtype=torch.float32),
    }

    raw = MagFormerArch._inference_raw(outputs, (1, 3, 8, 8), include_raw_tensors=True)

    assert torch.is_tensor(raw["pred_logits"])
    assert torch.is_tensor(raw["pred_masks"])


def test_export_inference_predictions_converts_to_numpy() -> None:
    raw = {
        "predictions": [
            {
                "image_id": 0,
                "scores": torch.tensor([0.9], dtype=torch.float32),
                "category_ids": torch.tensor([0], dtype=torch.long),
                "masks": torch.rand(1, 8, 8, dtype=torch.float32),
            }
        ],
        "pred_logits": torch.randn(1, 1, 2, dtype=torch.float32),
        "pred_masks": torch.randn(1, 1, 8, 8, dtype=torch.float32),
    }

    exported = MagFormerArch._export_inference_predictions(raw, include_raw_tensors=False)
    pred = exported["predictions"][0]

    assert isinstance(pred["scores"], np.ndarray)
    assert isinstance(pred["category_ids"], np.ndarray)
    assert isinstance(pred["masks"], np.ndarray)
    assert "pred_logits" not in exported
    assert "pred_masks" not in exported


def test_export_inference_predictions_can_keep_raw_tensors() -> None:
    raw = {
        "predictions": [
            {
                "image_id": 0,
                "scores": torch.tensor([0.9], dtype=torch.float32),
                "category_ids": torch.tensor([0], dtype=torch.long),
                "masks": torch.rand(1, 8, 8, dtype=torch.float32),
            }
        ],
        "pred_logits": torch.randn(1, 1, 2, dtype=torch.float32),
        "pred_masks": torch.randn(1, 1, 8, 8, dtype=torch.float32),
    }

    exported = MagFormerArch._export_inference_predictions(
        raw,
        include_raw_tensors=True,
        move_raw_tensors_to_cpu=False,
    )

    assert isinstance(exported["pred_logits"], torch.Tensor)
    assert isinstance(exported["pred_masks"], torch.Tensor)
