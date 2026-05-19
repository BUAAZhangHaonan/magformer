from __future__ import annotations

import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _r82():
    return importlib.import_module("tools.diagnose_r82_tta_candidate_bank")




class _ForwardMaskRecorder:
    def __init__(self) -> None:
        self.depth_valid_masks: list[torch.Tensor | None] = []

    def forward_inference_raw(self, images, depths, **kwargs):
        del depths
        mask = kwargs.get("depth_valid_masks")
        self.depth_valid_masks.append(None if mask is None else mask.detach().clone())
        height, width = images.shape[-2:]
        return {
            "predictions": [
                {
                    "scores": images.new_empty((0,)),
                    "category_ids": torch.empty((0,), dtype=torch.long, device=images.device),
                    "masks": images.new_zeros((0, height, width)),
                }
            ]
        }


class _RunModel:
    def to(self, device):
        del device
        return self

    def eval(self) -> None:
        return None


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        runtime=SimpleNamespace(seed=7),
        data=SimpleNamespace(
            image_size=4,
            depth=SimpleNamespace(
                scale=1.0,
                shift=0.0,
                clip_min=None,
                clip_max=None,
                norm="none",
                per_sample_norm=False,
            ),
        ),
    )


class _Dataset:
    def __init__(self) -> None:
        self.coco = SimpleNamespace(imgs={10: {"height": 4, "width": 4}})
        self.category_ids = [1]
        self.transform = None

    def __len__(self) -> int:
        return 1


def _candidate(r82):
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    return r82.Candidate(0.95, 1, [1.0, 1.0, 3.0, 3.0], mask, "fake")


def _install_run_fakes(monkeypatch, r82, tmp_path, run_view):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    depth_valid_masks = torch.tensor([[[[True, False], [False, True]]]])
    batch = {
        "image_ids": torch.tensor([10]),
        "images": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
        "depths": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "padding_masks": torch.zeros((1, 4, 4), dtype=torch.bool),
        "depth_valid_masks": depth_valid_masks,
        "content_masks": [np.ones((4, 4), dtype=np.uint8)],
    }
    monkeypatch.setattr(r82, "load_yaml_file", lambda path: {})
    monkeypatch.setattr(r82, "merge_configs", lambda base, overrides: overrides)
    monkeypatch.setattr(r82, "save_yaml_file", lambda data, path: Path(path).write_text("{}\n", encoding="utf-8"))
    monkeypatch.setattr(r82, "load_config", lambda path: _cfg())
    monkeypatch.setattr(r82, "set_seed", lambda seed: None)
    monkeypatch.setattr(r82, "CocoRgbdDataset", lambda **kwargs: _Dataset())
    monkeypatch.setattr(r82, "DataLoader", lambda *args, **kwargs: [batch])
    monkeypatch.setattr(r82, "build_model", lambda cfg: _RunModel())
    monkeypatch.setattr(r82, "strict_load_with_evidence", lambda model, path: None)
    monkeypatch.setattr(r82, "_load_gt_buckets", lambda path: ({10: {}}, 1.0))
    monkeypatch.setattr(r82, "_load_r78_stats", lambda path: {10: {"r78_bucket": "normal", "file_name": "x.png"}})
    monkeypatch.setattr(r82, "_run_view", run_view)
    monkeypatch.setattr(r82, "_view_candidates_to_gt", lambda *args, **kwargs: [_candidate(r82)])
    monkeypatch.setattr(r82, "summarize_candidate_bank_buckets", lambda *args, **kwargs: {"buckets": {}})
    monkeypatch.setattr(r82, "_evaluate_gates", lambda summary: {"passed": True})
    monkeypatch.setattr(r82, "write_markdown_report", lambda *args, **kwargs: None)
    return weights, depth_valid_masks


def _args(tmp_path, weights) -> SimpleNamespace:
    return SimpleNamespace(
        base_config="base.yaml",
        weights=str(weights),
        dataset_root="dataset",
        target_ann="annotations.json",
        split="train",
        r78_stats="r78.json",
        output_dir=str(tmp_path / "out"),
        max_images=1,
        image_size=4,
        batch_size=1,
        num_workers=0,
        device="cpu",
        amp=False,
        inference_topk=3,
        score_threshold=0.0,
        mask_threshold=0.5,
        kept_threshold=0.1,
        report_interval=1,
        bucket_json=None,
        summary_json=None,
        candidate_json=None,
        markdown=None,
    )


def test_run_view_transforms_and_passes_depth_valid_masks() -> None:
    r82 = _r82()
    model = _ForwardMaskRecorder()
    images = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    depths = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    padding_masks = torch.zeros((1, 2, 2), dtype=torch.bool)
    depth_valid_masks = torch.tensor([[[[True, False], [False, True]]]])

    r82._run_view(
        model,
        images,
        depths,
        padding_masks,
        depth_valid_masks=depth_valid_masks,
        scale=2.0,
        hflip=True,
        device=torch.device("cpu"),
        amp_enabled=False,
        inference_topk=5,
    )

    expected = torch.nn.functional.interpolate(depth_valid_masks.float(), size=(4, 4), mode="nearest").bool()
    expected = torch.flip(expected, [-1])
    assert len(model.depth_valid_masks) == 1
    assert torch.equal(model.depth_valid_masks[0], expected)


def test_r82_run_passes_batch_depth_valid_masks_to_each_tta_view(monkeypatch, tmp_path) -> None:
    r82 = _r82()
    seen_masks: list[torch.Tensor] = []

    def run_view(model, images, depths, padding_masks, *, depth_valid_masks, **kwargs):
        del model, images, depths, padding_masks, kwargs
        seen_masks.append(depth_valid_masks.detach().clone())
        return {"predictions": [{}]}

    weights, depth_valid_masks = _install_run_fakes(monkeypatch, r82, tmp_path, run_view)

    r82.run(_args(tmp_path, weights))

    assert len(seen_masks) == len(r82.TTA_VIEWS)
    assert all(torch.equal(mask, depth_valid_masks) for mask in seen_masks)


def test_union_clusters_same_class_by_bbox_iou_and_keeps_best_mask() -> None:
    r82 = _r82()
    mask_a = np.zeros((8, 8), dtype=np.uint8)
    mask_a[1:7, 1:7] = 1
    mask_b = np.zeros((8, 8), dtype=np.uint8)
    mask_b[1:4, 1:4] = 1
    mask_c = np.zeros((8, 8), dtype=np.uint8)
    mask_c[1:7, 1:7] = 1

    candidates = [
        r82.Candidate(0.70, 1, [1.0, 1.0, 7.0, 7.0], mask_a, "s1"),
        r82.Candidate(0.90, 1, [1.2, 1.1, 6.8, 6.9], mask_b, "s2"),
        r82.Candidate(0.95, 2, [1.0, 1.0, 7.0, 7.0], mask_c, "other-class"),
    ]

    merged = r82.union_cluster_candidates(
        candidates,
        mask_iou_threshold=0.55,
        bbox_iou_threshold=0.75,
    )

    assert len(merged) == 2
    class_one = next(item for item in merged if item.category_id == 1)
    assert class_one.quality == 0.90
    assert class_one.source_view == "s2"
    np.testing.assert_array_equal(class_one.mask, mask_b)


def test_bucket_summary_values_are_finite_for_tiny_and_bottom_rows() -> None:
    r82 = _r82()
    gt_by_image = {
        10: {
            "image": {"id": 10, "width": 8, "height": 8},
            "all": [
                {"id": 1, "image_id": 10, "bbox": [1, 1, 2, 2], "_area": 4.0},
            ],
            "tiny_area_le_256": [
                {"id": 1, "image_id": 10, "bbox": [1, 1, 2, 2], "_area": 4.0},
            ],
            "bottom20_area": [
                {"id": 1, "image_id": 10, "bbox": [1, 1, 2, 2], "_area": 4.0},
            ],
        }
    }
    r78_by_image = {10: {"r78_bucket": "normal", "file_name": "x.png"}}
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    record = {
        "image_id": 10,
        "target_size": (8, 8),
        "candidates": [r82.Candidate(0.2, 1, [1.0, 1.0, 3.0, 3.0], mask, "v")],
        "kept": [r82.Candidate(0.2, 1, [1.0, 1.0, 3.0, 3.0], mask, "v")],
    }

    summary = r82.summarize_candidate_bank_buckets(
        [record],
        gt_by_image=gt_by_image,
        r78_by_image=r78_by_image,
        bottom20_threshold=4.0,
        target_ann_path="ann.json",
        r78_stats_path="stats.json",
    )

    for bucket in summary["buckets"].values():
        for value in bucket.values():
            if isinstance(value, float):
                assert math.isfinite(value)

    assert summary["buckets"]["tiny_area_le_256"]["candidate_gt_coverage_iou50"] == 1.0
    assert summary["buckets"]["bottom20_area"]["kept_gt_coverage_iou75"] == 1.0
