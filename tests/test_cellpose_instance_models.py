from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
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


def _write_split(root: Path, split: str, image_id: int, *, size: int = 16) -> None:
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    image_name = f"{split}_{image_id:06d}.png"
    Image.new("RGB", (size, size), color=(12, 34, 56)).save(root / "images" / split / image_name)
    payload = {
        "images": [{"id": image_id, "file_name": image_name, "width": size, "height": size}],
        "annotations": [
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [[2, 2, 7, 2, 7, 7, 2, 7]],
                "area": 25,
                "bbox": [2, 2, 5, 5],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_min_dataset(root: Path) -> None:
    _write_split(root, "train", 1)
    _write_split(root, "val", 2)


def test_cellpose_official_dependency_is_required() -> None:
    mod = _load_module()

    assert mod.CELLPOSE_VERSION == "3.1.1.1"
    assert mod.CELLPOSE_MODEL_CLS.__name__ == "CellposeModel"
    assert callable(mod.CELLPOSE_TRAIN_SEG_FN)
    assert hasattr(mod.CELLPOSE_DYNAMICS, "masks_to_flows_gpu")
    assert hasattr(mod.CELLPOSE_DYNAMICS, "compute_masks")


def test_cellpose_targets_use_official_diffusion_flows(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    calls: list[tuple[np.ndarray, Any]] = []

    def fake_masks_to_flows_gpu(mask: np.ndarray, device=None, niter=None):
        calls.append((mask.copy(), device))
        return np.ones((2, mask.shape[0], mask.shape[1]), dtype=np.float32), np.zeros((2, 2), dtype=np.int32)

    monkeypatch.setattr(mod.CELLPOSE_DYNAMICS, "masks_to_flows_gpu", fake_masks_to_flows_gpu)
    instance_map = np.zeros((8, 8), dtype=np.int32)
    instance_map[2:6, 2:6] = 1

    targets = mod.instance_map_to_cellpose_targets(instance_map)

    assert len(calls) == 1
    assert np.array_equal(calls[0][0], instance_map)
    assert targets["flow"].shape == (2, 8, 8)
    assert np.allclose(targets["flow"], 1.0)
    assert targets["cellprob"].sum() == 16


def test_cellpose_label_map_adapter_exports_scores_from_cellprob() -> None:
    mod = _load_module()
    label_map = np.zeros((12, 12), dtype=np.int32)
    label_map[2:6, 2:6] = 1
    label_map[7:10, 7:11] = 2
    cellprob = np.full((12, 12), -4.0, dtype=np.float32)
    cellprob[label_map == 1] = 4.0
    cellprob[label_map == 2] = 2.0

    masks, scores, category_ids = mod.label_map_to_instance_predictions(
        label_map,
        cellprob=cellprob,
        min_area=4,
    )

    assert len(masks) == 2
    assert scores.shape == (2,)
    assert scores[0] > scores[1] > 0.5
    assert category_ids.tolist() == [0, 0]


def test_train_cellpose_model_calls_official_train_seg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    dataset_root = tmp_path / "ecc"
    output_dir = tmp_path / "out"
    _write_min_dataset(dataset_root)
    calls: dict[str, Any] = {}

    class FakeNet:
        device = "cpu"

        def save_model(self, filename):
            Path(filename).write_bytes(b"fake-cellpose")

        def parameters(self):
            return []

    class FakeCellposeModel:
        def __init__(self, **kwargs):
            calls["model_kwargs"] = kwargs
            self.net = FakeNet()

    def fake_train_seg(net, **kwargs):
        calls["train_net"] = net
        calls["train_kwargs"] = kwargs
        filename = Path(kwargs["save_path"]) / "models" / kwargs["model_name"]
        filename.parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"trained")
        return str(filename), [1.0], [1.0]

    monkeypatch.setattr(mod, "CELLPOSE_MODEL_CLS", FakeCellposeModel)
    monkeypatch.setattr(mod, "CELLPOSE_TRAIN_SEG_FN", fake_train_seg)

    bundle = mod.train_cellpose_model(
        dataset_root=dataset_root,
        output_dir=output_dir,
        image_size=16,
        epochs=3,
        batch=2,
        lr=0.01,
        num_workers=0,
        device="cpu",
        max_train_steps=2,
    )

    assert calls["model_kwargs"]["pretrained_model"] is False
    assert calls["model_kwargs"]["nchan"] == 3
    assert calls["train_kwargs"]["compute_flows"] is True
    assert calls["train_kwargs"]["channel_axis"] is None
    assert calls["train_kwargs"]["batch_size"] == 2
    assert calls["train_kwargs"]["n_epochs"] == 2
    assert bundle["checkpoint"].name == "model_final.pth"
    assert bundle["checkpoint"].is_file()
    assert bundle["checkpoint"].parent == output_dir


def test_predict_records_calls_official_cellpose_eval(tmp_path: Path) -> None:
    mod = _load_module()
    dataset_root = tmp_path / "ecc"
    _write_min_dataset(dataset_root)
    calls: list[dict[str, Any]] = []

    class FakeOfficialModel:
        def eval(self, images, **kwargs):
            calls.append({"num_images": len(images), **kwargs})
            masks = []
            flows = []
            for _image in images:
                label_map = np.zeros((16, 16), dtype=np.int32)
                label_map[4:8, 4:8] = 1
                masks.append(label_map)
                flows.append([None, None, np.full((16, 16), 3.0, dtype=np.float32)])
            return masks, flows, None

    rows = mod.predict_records(
        model_bundle={"model": FakeOfficialModel()},
        dataset_root=dataset_root,
        eval_split="val",
        image_size=16,
        min_area=4,
        device="cpu",
        inference_batch_size=2,
    )

    assert calls
    assert calls[0]["channel_axis"] == 0
    assert calls[0]["compute_masks"] is True
    assert calls[0]["resample"] is True
    assert calls[0]["flow_threshold"] == pytest.approx(0.4)
    assert calls[0]["cellprob_threshold"] == pytest.approx(0.0)
    assert len(rows) == 1
    assert rows[0]["bbox"] == [4.0, 4.0, 4.0, 4.0]
    assert rows[0]["score"] > 0.9


def test_precompute_cellpose_target_cache_writes_official_flow_entries(tmp_path: Path) -> None:
    mod = _load_module()
    dataset_root = tmp_path / "ecc"
    _write_min_dataset(dataset_root)
    cache_dir = tmp_path / "cache"

    summary = mod.precompute_cellpose_target_cache(
        dataset_root=dataset_root,
        split="train",
        image_size=16,
        target_cache_dir=cache_dir,
        num_workers=0,
    )

    assert summary["records"] == 1
    assert summary["created"] == 1
    assert summary["cache_version"] == mod.CELLPOSE_TARGET_CACHE_VERSION
    cache_files = list(cache_dir.glob("*.npz"))
    assert len(cache_files) == 1
    with np.load(cache_files[0]) as payload:
        assert payload["flow"].shape == (2, 16, 16)
        assert payload["instance_map"].shape == (16, 16)


def test_evaluate_results_serializes_mask_rows_for_coco_eval(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools")
    mod = _load_module()
    dataset_root = tmp_path / "ecc"
    _write_split(dataset_root, "val", 1)

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
