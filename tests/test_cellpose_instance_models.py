from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "baselines" / "cellpose_instance_models.py"
    spec = importlib.util.spec_from_file_location("cellpose_instance_models", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _matched_iou_stats(pred_masks: list[np.ndarray], instance_map: np.ndarray) -> tuple[int, float]:
    gt_masks = [(instance_map == instance_id).astype(np.uint8) for instance_id in np.unique(instance_map) if instance_id > 0]
    if not pred_masks or not gt_masks:
        return 0, 0.0
    ious = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for pred_idx, pred in enumerate(pred_masks):
        pred_bool = pred.astype(bool)
        for gt_idx, gt in enumerate(gt_masks):
            gt_bool = gt.astype(bool)
            inter = np.logical_and(pred_bool, gt_bool).sum()
            union = np.logical_or(pred_bool, gt_bool).sum()
            ious[pred_idx, gt_idx] = float(inter / max(1, union))
    matched: list[float] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for _ in range(min(len(pred_masks), len(gt_masks))):
        best = np.unravel_index(np.argmax(ious), ious.shape)
        pred_idx, gt_idx = int(best[0]), int(best[1])
        if ious[pred_idx, gt_idx] < 0:
            break
        matched.append(float(ious[pred_idx, gt_idx]))
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        ious[pred_idx, :] = -1.0
        ious[:, gt_idx] = -1.0
    return len(matched), float(np.mean(matched)) if matched else 0.0


def _oracle_predictions(mod, instance_map: np.ndarray, *, min_area: int = 4) -> list[np.ndarray]:
    targets = mod.instance_map_to_cellpose_targets(instance_map)
    masks, _scores, _category_ids = mod.predictions_from_logits(
        flow_logits=targets["flow"] * mod.CELLPOSE_FLOW_LOGIT_SCALE,
        cellprob_logits=np.where(targets["cellprob"] > 0, 8.0, -8.0).astype(np.float32),
        min_area=min_area,
        score_threshold=0.05,
        mask_threshold=0.5,
    )
    return masks


def test_cellpose_oracle_round_trip_recovers_irregular_and_touching_instances() -> None:
    mod = _load_module()
    yy, xx = np.mgrid[:64, :64]
    instance_map = np.zeros((64, 64), dtype=np.int32)
    instance_map[((yy - 16) ** 2 + (xx - 15) ** 2) <= 9**2] = 1
    instance_map[8:26, 28:38] = 2
    instance_map[28:45, 8:24] = 3
    instance_map[35:45, 18:32] = 3  # concave-ish L/step shape
    instance_map[30:48, 32:44] = 4  # touches instance 3 along one edge
    ring = ((yy - 47) ** 2 + (xx - 16) ** 2 <= 9**2) & ((yy - 47) ** 2 + (xx - 16) ** 2 >= 4**2)
    instance_map[ring] = 5

    masks = _oracle_predictions(mod, instance_map, min_area=12)
    matched_count, mean_iou = _matched_iou_stats(masks, instance_map)

    assert len(masks) == 5
    assert matched_count == 5
    assert mean_iou >= 0.95


def test_cellpose_flow_logit_scale_is_symmetric_for_training_and_inference() -> None:
    mod = _load_module()
    instance_map = np.zeros((32, 32), dtype=np.int32)
    instance_map[4:16, 4:14] = 1
    instance_map[12:24, 14:25] = 2
    targets = mod.instance_map_to_cellpose_targets(instance_map)

    scaled_masks, _scores, _category_ids = mod.predictions_from_logits(
        flow_logits=targets["flow"] * mod.CELLPOSE_FLOW_LOGIT_SCALE,
        cellprob_logits=np.where(targets["cellprob"] > 0, 8.0, -8.0).astype(np.float32),
        min_area=8,
        score_threshold=0.05,
        mask_threshold=0.5,
    )
    unscaled_masks, _scores, _category_ids = mod.predictions_from_logits(
        flow_logits=targets["flow"],
        cellprob_logits=np.where(targets["cellprob"] > 0, 8.0, -8.0).astype(np.float32),
        min_area=8,
        score_threshold=0.05,
        mask_threshold=0.5,
        flow_logit_scale=1.0,
    )

    assert _matched_iou_stats(scaled_masks, instance_map) == pytest.approx(_matched_iou_stats(unscaled_masks, instance_map))


def test_cellpose_oracle_predictions_score_high_in_mini_coco_eval(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools")
    from pycocotools import mask as mask_utils

    mod = _load_module()
    dataset_root = tmp_path / "ecc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    image_name = "val_000001.png"
    Image.new("RGB", (32, 32), color=(12, 34, 56)).save(dataset_root / "images" / "val" / image_name)

    instance_map = np.zeros((32, 32), dtype=np.int32)
    instance_map[4:18, 4:15] = 1
    instance_map[10:24, 15:26] = 2
    annotations = []
    for instance_id in [1, 2]:
        mask = (instance_map == instance_id).astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        ys, xs = np.nonzero(mask)
        annotations.append(
            {
                "id": instance_id,
                "image_id": 1,
                "category_id": 1,
                "segmentation": rle,
                "area": int(mask.sum()),
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                "iscrowd": 0,
            }
        )
    payload = {
        "images": [{"id": 1, "file_name": image_name, "width": 32, "height": 32}],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "component"}],
    }
    (dataset_root / "annotations" / "instances_val.json").write_text(json.dumps(payload), encoding="utf-8")

    masks = _oracle_predictions(mod, instance_map, min_area=8)
    rows = mod.binary_masks_to_coco_rows(
        image_id=1,
        masks=masks,
        scores=np.ones((len(masks),), dtype=np.float32),
        category_ids=np.zeros((len(masks),), dtype=np.int64),
    )
    metrics = mod.evaluate_results(dataset_root=dataset_root, eval_split="val", rows=rows, image_size=32, iteration=1)

    assert metrics["segm/AP"] >= 99.0
    assert metrics["segm/AP50"] >= 99.0


def test_instance_map_to_cellpose_targets_produces_flow_and_cellprob() -> None:
    mod = _load_module()
    instance_map = np.zeros((16, 16), dtype=np.int32)
    instance_map[2:6, 2:6] = 1
    instance_map[9:13, 9:14] = 2

    targets = mod.instance_map_to_cellpose_targets(instance_map)

    assert targets["cellprob"].shape == (16, 16)
    assert targets["flow"].shape == (2, 16, 16)
    assert targets["cellprob"].dtype == np.float32
    assert targets["flow"].dtype == np.float32
    assert targets["cellprob"][3, 3] == 1.0
    assert targets["cellprob"][0, 0] == 0.0
    assert np.any(np.abs(targets["flow"]) > 0.0)
    assert targets["flow"][1, 3, 2] > 0.0
    assert targets["flow"][1, 3, 5] < 0.0
    assert targets["flow"][0, 2, 3] > 0.0
    assert targets["flow"][0, 5, 3] < 0.0


def test_cellpose_dataset_uses_lightweight_records(tmp_path: Path) -> None:
    mod = _load_module()
    dataset_root = tmp_path / "ecc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    image_name = "train_000001.png"
    Image.new("RGB", (16, 16), color=(12, 34, 56)).save(dataset_root / "images" / "train" / image_name)
    payload = {
        "images": [{"id": 1, "file_name": image_name, "width": 16, "height": 16}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[2, 2, 7, 2, 7, 7, 2, 7]],
                "area": 25,
                "bbox": [2, 2, 5, 5],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (dataset_root / "annotations" / "instances_train.json").write_text(json.dumps(payload), encoding="utf-8")

    dataset = mod.ECCCellPoseDataset(dataset_root, "train", image_size=16, train=False)
    assert "annotation_targets" not in dataset.records[0]
    sample = dataset[0]
    assert sample["instance_map"].shape == (16, 16)
    assert sample["cellprob"].shape == (1, 16, 16)


def test_cellpose_dataset_reuses_disk_cached_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    dataset_root = tmp_path / "ecc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    image_name = "train_000001.png"
    Image.new("RGB", (16, 16), color=(12, 34, 56)).save(dataset_root / "images" / "train" / image_name)
    payload = {
        "images": [{"id": 1, "file_name": image_name, "width": 16, "height": 16}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[2, 2, 7, 2, 7, 7, 2, 7]],
                "area": 25,
                "bbox": [2, 2, 5, 5],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (dataset_root / "annotations" / "instances_train.json").write_text(json.dumps(payload), encoding="utf-8")

    cache_dir = tmp_path / "cache"
    dataset = mod.ECCCellPoseDataset(dataset_root, "train", image_size=16, train=False, target_cache_dir=cache_dir)
    first = dataset[0]
    assert list(cache_dir.glob("*.npz"))

    def fail_if_recomputed(_instance_map):
        raise AssertionError("cached target was recomputed")

    monkeypatch.setattr(mod, "instance_map_to_cellpose_targets", fail_if_recomputed)
    cached_dataset = mod.ECCCellPoseDataset(dataset_root, "train", image_size=16, train=False, target_cache_dir=cache_dir)
    second = cached_dataset[0]
    assert np.array_equal(first["instance_map"].numpy(), second["instance_map"].numpy())
    assert np.array_equal(first["cellprob"].numpy(), second["cellprob"].numpy())


def test_precompute_cellpose_target_cache_writes_expected_entries(tmp_path: Path) -> None:
    mod = _load_module()
    dataset_root = tmp_path / "ecc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    for image_id in [1, 2]:
        image_name = f"train_{image_id:06d}.png"
        Image.new("RGB", (16, 16), color=(12, 34, 56)).save(dataset_root / "images" / "train" / image_name)
        images.append({"id": image_id, "file_name": image_name, "width": 16, "height": 16})
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [[2, 2, 7, 2, 7, 7, 2, 7]],
                "area": 25,
                "bbox": [2, 2, 5, 5],
                "iscrowd": 0,
            }
        )
    payload = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "component"}]}
    (dataset_root / "annotations" / "instances_train.json").write_text(json.dumps(payload), encoding="utf-8")

    cache_dir = tmp_path / "cache"
    summary = mod.precompute_cellpose_target_cache(
        dataset_root=dataset_root,
        split="train",
        image_size=16,
        target_cache_dir=cache_dir,
        num_workers=0,
    )

    assert summary["records"] == 2
    assert summary["created"] == 2
    assert summary["existing"] == 0
    assert len(list(cache_dir.glob("*.npz"))) == 2
    with np.load(next(cache_dir.glob("*.npz"))) as payload:
        assert str(payload["cache_version"].item()) == mod.CELLPOSE_TARGET_CACHE_VERSION


def test_cellpose_target_cache_rejects_stale_version(tmp_path: Path) -> None:
    mod = _load_module()
    cache_path = tmp_path / "stale.npz"
    np.savez(
        cache_path,
        instance_map=np.zeros((8, 8), dtype=np.uint16),
        cellprob=np.zeros((8, 8), dtype=np.uint8),
        flow=np.zeros((2, 8, 8), dtype=np.float16),
        cache_version=np.asarray("flow-v2"),
    )

    assert mod._read_cached_cellpose_targets(cache_path) is None


def test_follow_flows_and_scores_round_trip_separates_instances() -> None:
    mod = _load_module()
    instance_map = np.zeros((24, 24), dtype=np.int32)
    instance_map[3:9, 3:9] = 1
    instance_map[14:20, 14:21] = 2
    targets = mod.instance_map_to_cellpose_targets(instance_map)

    masks, scores, category_ids = mod.predictions_from_logits(
        flow_logits=targets["flow"] * 5.0,
        cellprob_logits=np.where(targets["cellprob"] > 0, 8.0, -8.0).astype(np.float32),
        min_area=5,
        score_threshold=0.05,
        mask_threshold=0.5,
    )

    assert len(masks) == 2
    assert scores.shape == (2,)
    assert category_ids.tolist() == [0, 0]
    assert all(mask.dtype == np.uint8 for mask in masks)
    assert scores[0] <= 1.0 and scores[1] <= 1.0
    assert scores[0] >= 0.5 and scores[1] >= 0.5


def test_cellpose_prediction_rows_resize_masks_to_original_record_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(12, 34, 56)).save(image_path)
    record = {
        "image_id": 1,
        "image_path": str(image_path),
        "file_name": image_path.name,
        "height": 32,
        "width": 32,
        "annotations": [],
    }
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1

    class FakeModel:
        def __call__(self, _image):
            return torch.zeros((1, 3, 16, 16), dtype=torch.float32)

    monkeypatch.setattr(
        mod,
        "predictions_from_logits",
        lambda **_kwargs: ([mask], np.asarray([0.9], dtype=np.float32), np.asarray([0], dtype=np.int64)),
    )
    rows = mod._predict_rows_for_record(
        model=FakeModel(),
        record=record,
        image_size=16,
        min_area=1,
        device="cpu",
        score_threshold=0.05,
        mask_threshold=0.5,
    )

    assert len(rows) == 1
    assert rows[0]["bbox"] == [8.0, 8.0, 8.0, 8.0]


def test_cellpose_predict_records_batches_model_forward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    dataset_root = tmp_path / "ecc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    images = []
    for image_id in [1, 2]:
        image_name = f"val_{image_id:06d}.png"
        Image.new("RGB", (16, 16), color=(12, 34, 56)).save(dataset_root / "images" / "val" / image_name)
        images.append({"id": image_id, "file_name": image_name, "width": 16, "height": 16})
    payload = {"images": images, "annotations": [], "categories": [{"id": 1, "name": "component"}]}
    (dataset_root / "annotations" / "instances_val.json").write_text(json.dumps(payload), encoding="utf-8")

    calls: list[int] = []

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, image_batch):
            calls.append(int(image_batch.shape[0]))
            return torch.zeros((image_batch.shape[0], 3, 16, 16), dtype=torch.float32)

    monkeypatch.setattr(
        mod,
        "predictions_from_logits",
        lambda **_kwargs: ([], np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)),
    )

    rows = mod.predict_records(
        model_bundle={"model": FakeModel()},
        dataset_root=dataset_root,
        eval_split="val",
        image_size=16,
        min_area=1,
        device="cpu",
        inference_batch_size=2,
    )

    assert rows == []
    assert calls == [2]


def test_evaluate_results_serializes_mask_rows_for_coco_eval(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools")

    mod = _load_module()
    dataset_root = tmp_path / "ecc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)

    image_name = "val_000001.png"
    Image.new("RGB", (16, 16), color=(12, 34, 56)).save(dataset_root / "images" / "val" / image_name)
    payload = {
        "images": [{"id": 1, "file_name": image_name, "width": 16, "height": 16}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[2, 2, 7, 2, 7, 7, 2, 7]],
                "area": 25,
                "bbox": [2, 2, 5, 5],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (dataset_root / "annotations" / "instances_val.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = [
        {
            "image_id": 1,
            "category_id": 1,
            "score": 0.9,
            "bbox": [2.0, 2.0, 5.0, 5.0],
            "mask": np.pad(np.ones((5, 5), dtype=np.uint8), ((2, 9), (2, 9))),
        }
    ]
    metrics = mod.evaluate_results(
        dataset_root=dataset_root,
        eval_split="val",
        rows=rows,
        image_size=16,
        iteration=1,
    )

    assert "segm/AP" in metrics
