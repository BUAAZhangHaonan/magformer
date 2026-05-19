from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from pycocotools import mask as coco_mask


def _r83():
    return importlib.import_module("tools.build_r83_tta_coco_bank")




class _RunModel:
    def to(self, device):
        del device
        return self

    def eval(self) -> None:
        return None


class _Dataset:
    def __init__(self) -> None:
        self.coco = SimpleNamespace(imgs={10: {"height": 4, "width": 4}})
        self.category_ids = [1]
        self.transform = None

    def __len__(self) -> int:
        return 1


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


def test_r83_run_passes_batch_depth_valid_masks_to_each_tta_view(monkeypatch, tmp_path) -> None:
    r82 = importlib.import_module("tools.diagnose_r82_tta_candidate_bank")
    r83 = _r83()
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
    seen_masks: list[torch.Tensor] = []

    def run_view(model, images, depths, padding_masks, *, depth_valid_masks, **kwargs):
        del model, images, depths, padding_masks, kwargs
        seen_masks.append(depth_valid_masks.detach().clone())
        return {"predictions": [{}]}

    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    candidate = r82.Candidate(0.95, 1, [1.0, 1.0, 3.0, 3.0], mask, "fake")
    monkeypatch.setattr(r83, "load_yaml_file", lambda path: {})
    monkeypatch.setattr(r83, "merge_configs", lambda base, overrides: overrides)
    monkeypatch.setattr(r83, "save_yaml_file", lambda data, path: Path(path).write_text("{}\n", encoding="utf-8"))
    monkeypatch.setattr(r83, "load_config", lambda path: _cfg())
    monkeypatch.setattr(r83, "set_seed", lambda seed: None)
    monkeypatch.setattr(r83, "CocoRgbdDataset", lambda **kwargs: _Dataset())
    monkeypatch.setattr(r83, "DataLoader", lambda *args, **kwargs: [batch])
    monkeypatch.setattr(r83, "build_model", lambda cfg: _RunModel())
    monkeypatch.setattr(r83, "strict_load_with_evidence", lambda model, path: None)
    monkeypatch.setattr(r83, "_load_gt_buckets", lambda path: ({10: {}}, 1.0))
    monkeypatch.setattr(r83, "_load_r78_stats", lambda path: {10: {"r78_bucket": "normal", "file_name": "x.png"}})
    monkeypatch.setattr(r83, "_run_view", run_view)
    monkeypatch.setattr(r83, "_view_candidates_to_gt", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(r83, "union_cluster_candidates", lambda candidates: candidates[:1])
    monkeypatch.setattr(r83, "_load_target_coco_subset", lambda path, image_ids: {"images": [{"id": 10, "file_name": "x.png", "height": 4, "width": 4}], "annotations": [], "categories": [{"id": 1, "name": "object"}]})
    monkeypatch.setattr(r83, "summarize_candidate_bank_buckets", lambda *args, **kwargs: {"buckets": {}})
    monkeypatch.setattr(r83, "validate_coco_bank", lambda *args, **kwargs: {"images": 1, "annotations": 1})

    args = SimpleNamespace(
        base_config="base.yaml",
        weights=str(weights),
        dataset_root="dataset",
        target_ann="annotations.json",
        split="train",
        r78_stats="r78.json",
        output_dir=str(tmp_path / "out"),
        output_json=None,
        max_images=1,
        image_size=4,
        batch_size=1,
        num_workers=0,
        device="cpu",
        amp=False,
        inference_topk=3,
        score_threshold=0.0,
        mask_threshold=0.5,
        train_score_threshold=0.9,
        min_mask_area=1,
        min_fill_ratio=0.0,
        report_interval=1,
        r82_candidate_json="missing.json",
        validate_only=False,
        allow_empty_images=False,
    )

    r83.run(args)

    assert len(seen_masks) == len(r83.TTA_VIEWS)
    assert all(torch.equal(mask, depth_valid_masks) for mask in seen_masks)


def test_candidate_to_coco_annotation_writes_valid_rle_and_xywh_bbox() -> None:
    r82 = importlib.import_module("tools.diagnose_r82_tta_candidate_bank")
    r83 = _r83()
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3:7] = 1
    candidate = r82.Candidate(
        quality=0.95,
        category_id=1,
        bbox=[3.0, 2.0, 7.0, 6.0],
        mask=mask,
        source_view="scale1_noflip",
    )

    annotation = r83.candidate_to_coco_annotation(
        candidate,
        annotation_id=11,
        image_id=123,
        image_height=8,
        image_width=8,
    )

    assert annotation["id"] == 11
    assert annotation["image_id"] == 123
    assert annotation["category_id"] == 1
    assert annotation["bbox"] == [3.0, 2.0, 4.0, 4.0]
    assert annotation["area"] == 16
    assert annotation["score"] == 0.95
    assert annotation["iscrowd"] == 0
    decoded = coco_mask.decode(annotation["segmentation"])
    np.testing.assert_array_equal(decoded, mask)


def test_select_training_candidates_applies_score_area_and_fill_filters() -> None:
    r82 = importlib.import_module("tools.diagnose_r82_tta_candidate_bank")
    r83 = _r83()
    good = np.zeros((12, 12), dtype=np.uint8)
    good[1:7, 1:7] = 1
    tiny = np.zeros((12, 12), dtype=np.uint8)
    tiny[1:4, 1:4] = 1
    sparse = np.zeros((20, 20), dtype=np.uint8)
    sparse[1:6, 1:6] = 1
    low_score = good.copy()

    candidates = [
        r82.Candidate(0.95, 1, [1.0, 1.0, 7.0, 7.0], good, "good"),
        r82.Candidate(0.99, 1, [1.0, 1.0, 4.0, 4.0], tiny, "tiny"),
        r82.Candidate(0.99, 1, [0.0, 0.0, 20.0, 20.0], sparse, "sparse"),
        r82.Candidate(0.89, 1, [1.0, 1.0, 7.0, 7.0], low_score, "low"),
    ]

    kept, stats = r83.select_training_candidates(
        candidates,
        min_score=0.90,
        min_mask_area=20,
        min_fill_ratio=0.1,
    )

    assert [item.source_view for item in kept] == ["good"]
    assert stats == {
        "input": 4,
        "kept": 1,
        "dropped_score": 1,
        "dropped_mask_area": 1,
        "dropped_fill_ratio": 1,
    }
