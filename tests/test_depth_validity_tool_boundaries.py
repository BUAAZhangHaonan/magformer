from __future__ import annotations

from pathlib import Path

import pytest
import torch


class _Eval1024Recorder:
    def __init__(self) -> None:
        self.depth_valid_masks = None

    def forward_inference_raw(self, images, depths, **kwargs):
        del images, depths
        self.depth_valid_masks = kwargs.get("depth_valid_masks")
        return {"predictions": []}


def test_evaluate_1024_forward_batch_passes_depth_valid_masks() -> None:
    from tools import evaluate_1024_backmap

    model = _Eval1024Recorder()
    mask = torch.tensor([[[[True, False], [True, True]]]])
    batch = {
        "images": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        "depths": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        "depth_valid_masks": mask,
    }

    evaluate_1024_backmap.forward_raw_batch(
        model,
        batch,
        device=torch.device("cpu"),
        inference_topk=17,
    )

    assert model.depth_valid_masks is mask


class _TtaMaskRecorder:
    def __init__(self) -> None:
        self.depth_valid_masks: list[torch.Tensor | None] = []

    def forward_inference_raw(
        self,
        images,
        depths,
        *,
        depth_valid_masks=None,
        include_raw_tensors=False,
        move_predictions_to_cpu=True,
    ):
        del depths, include_raw_tensors, move_predictions_to_cpu
        self.depth_valid_masks.append(
            None if depth_valid_masks is None else depth_valid_masks.detach().clone()
        )
        height, width = images.shape[-2:]
        return {
            "predictions": [
                {
                    "scores": images.new_empty((0,)),
                    "category_ids": torch.empty((0,), dtype=torch.long, device=images.device),
                    "masks": images.new_zeros((0, height, width)),
                    "mask_probs": images.new_zeros((0, height, width)),
                }
            ]
        }


def test_tta_transforms_depth_valid_masks_with_nearest_and_passes_each_aug() -> None:
    from tools import evaluate_tta

    model = _TtaMaskRecorder()
    images = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    depths = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    mask = torch.tensor([[[[True, False], [False, True]]]])

    evaluate_tta.tta_inference_single_image(
        model,
        images,
        depths,
        [2.0],
        True,
        torch.device("cpu"),
        False,
        nms_iou=0.5,
        max_preds=10,
        score_thresh=0.0,
        original_h=2,
        original_w=2,
        mask_threshold=0.5,
        depth_valid_masks=mask,
    )

    expected = torch.nn.functional.interpolate(mask.float(), size=(4, 4), mode="nearest").bool()
    assert len(model.depth_valid_masks) == 2
    assert torch.equal(model.depth_valid_masks[0], expected)
    assert torch.equal(model.depth_valid_masks[1], torch.flip(expected, [-1]))


def test_depth_prior_extractor_missing_mask_fails_for_direct_valid_hole_output() -> None:
    from magformer.models.magformer.fusion import DepthPriorExtractor

    extractor = DepthPriorExtractor(use_rgb_edge=False)
    normalized_depth = torch.tensor([[[[0.0, 0.5], [1.0, 0.2]]]], dtype=torch.float32)

    with pytest.raises(ValueError, match="depth_valid_masks"):
        extractor(normalized_depth)


def test_inference_cli_constructs_and_passes_depth_valid_mask() -> None:
    script = Path(__file__).resolve().parents[1] / "tools" / "inference.py"
    text = script.read_text(encoding="utf-8")

    assert '"depth_valid_mask"' in text
    assert "depth_valid_masks=depth_valid_masks" in text
    assert "model.forward_inference_raw(images, depths)" not in text


def test_visualize_results_passes_depth_valid_masks() -> None:
    script = Path(__file__).resolve().parents[1] / "tools" / "visualize_results.py"
    text = script.read_text(encoding="utf-8")

    assert "batch.get(\"depth_valid_masks\")" in text
    assert "depth_valid_masks=depth_valid_masks" in text
    assert "model.forward_inference_raw(images, depths)" not in text


def test_benchmark_inference_decoder_forward_passes_depth_valid_masks() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "benchmark_inference.py"
    text = script.read_text(encoding="utf-8")

    assert "batch.get(\"depth_valid_masks\")" in text
    assert "depth_valid_masks=depth_valid_masks" in text
