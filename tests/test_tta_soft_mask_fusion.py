from __future__ import annotations

import importlib

import torch


def _planned_merge_soft_mask_clusters():
    tta_module = importlib.import_module("tools.evaluate_tta")
    return getattr(tta_module, "merge_soft_mask_clusters")


class _SoftMaskTtaModel:
    def __init__(self) -> None:
        self.include_raw_tensors_values = []

    def forward_inference_raw(self, images, depths, include_raw_tensors=False):
        del depths
        self.include_raw_tensors_values.append(include_raw_tensors)
        height, width = images.shape[-2:]
        soft_mask = images.new_full((1, height, width), 0.25)
        soft_mask[:, 1:3, 1:3] = 0.75
        binary_mask = images.new_ones((1, height, width))
        return {
            "predictions": [
                {
                    "scores": images.new_tensor([0.9]),
                    "category_ids": torch.tensor([0], dtype=torch.long, device=images.device),
                    "masks": binary_mask,
                    "mask_probs": soft_mask,
                }
            ]
        }


def test_tta_requests_raw_tensors_and_prefers_mask_probabilities() -> None:
    tta_module = importlib.import_module("tools.evaluate_tta")
    model = _SoftMaskTtaModel()
    images = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    depths = torch.zeros((1, 1, 4, 4), dtype=torch.float32)

    merged = tta_module.tta_inference_single_image(
        model, images, depths, [1.0], False, torch.device("cpu"), False,
        nms_iou=0.5, max_preds=10, score_thresh=0.0,
        original_h=4, original_w=4, mask_threshold=0.5,
    )

    expected_mask = torch.full((4, 4), 0.25, dtype=torch.float32)
    expected_mask[1:3, 1:3] = 0.75
    assert model.include_raw_tensors_values == [True]
    torch.testing.assert_close(merged["masks"][0], expected_mask)


def test_merge_soft_mask_clusters_score_weights_soft_masks_before_threshold() -> None:
    merge_soft_mask_clusters = _planned_merge_soft_mask_clusters()

    noisy_high_score = torch.zeros((4, 4), dtype=torch.float32)
    noisy_high_score[1:3, 1:3] = 0.60
    noisy_high_score[0, 0] = 0.70

    stable_lower_score = torch.zeros((4, 4), dtype=torch.float32)
    stable_lower_score[1:3, 1:3] = 0.80
    stable_lower_score[0, 0] = 0.10

    scores = torch.tensor([0.95, 0.85], dtype=torch.float32)
    masks = torch.stack([noisy_high_score, stable_lower_score])
    category_ids = torch.tensor([0, 0], dtype=torch.long)

    merged = merge_soft_mask_clusters(
        scores=scores,
        masks=masks,
        category_ids=category_ids,
        iou_threshold=0.5,
        mask_threshold=0.5,
        max_preds=10,
    )

    expected_soft = (
        scores[0] * noisy_high_score + scores[1] * stable_lower_score
    ) / scores.sum()
    expected_binary = torch.zeros((4, 4), dtype=torch.uint8)
    expected_binary[1:3, 1:3] = 1

    assert merged["scores"].shape == (1,)
    assert merged["category_ids"].tolist() == [0]
    torch.testing.assert_close(merged["masks"][0], expected_soft)

    thresholded = (merged["masks"][0] > 0.5).to(torch.uint8)
    torch.testing.assert_close(thresholded, expected_binary)

    first_binary = (noisy_high_score > 0.5).to(torch.uint8)
    assert first_binary[0, 0].item() == 1
    assert thresholded[0, 0].item() == 0


def test_nms_merge_uses_soft_mask_fusion_for_overlapping_segments() -> None:
    tta_module = importlib.import_module("tools.evaluate_tta")

    noisy_high_score = torch.zeros((4, 4), dtype=torch.float32)
    noisy_high_score[1:3, 1:3] = 0.60
    noisy_high_score[0, 0] = 0.70

    stable_lower_score = torch.zeros((4, 4), dtype=torch.float32)
    stable_lower_score[1:3, 1:3] = 0.80
    stable_lower_score[0, 0] = 0.10

    scores = torch.tensor([0.95, 0.85], dtype=torch.float32)
    masks = torch.stack([noisy_high_score, stable_lower_score])
    category_ids = torch.tensor([0, 0], dtype=torch.long)

    merged = tta_module.nms_merge(
        scores,
        masks,
        category_ids,
        iou_threshold=0.5,
        mask_threshold=0.5,
        max_preds=10,
    )

    expected_soft = (
        scores[0] * noisy_high_score + scores[1] * stable_lower_score
    ) / scores.sum()

    assert merged["scores"].shape == (1,)
    assert merged["category_ids"].tolist() == [0]
    torch.testing.assert_close(merged["masks"][0], expected_soft)
