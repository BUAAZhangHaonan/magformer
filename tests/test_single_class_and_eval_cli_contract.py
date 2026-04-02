from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from magformer.config import load_config
from magformer.config.validation import validate_config
from magformer.data import CocoRgbdDataset


def _write_dataset(root: Path, *, category_ids: list[int]) -> Path:
    (root / "images" / "train").mkdir(parents=True)
    (root / "depth" / "depth_npy" / "train").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)

    import cv2

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "images" / "train" / "0001.png"), image)
    np.save(root / "depth" / "depth_npy" / "train" / "0001.npy", np.zeros((16, 16), dtype=np.float32))

    annotations = []
    for ann_id, category_id in enumerate(category_ids, start=1):
        annotations.append(
            {
                "id": ann_id,
                "image_id": 1,
                "category_id": category_id,
                "bbox": [2, 2, 4, 4],
                "area": 16,
                "iscrowd": 0,
                "segmentation": [[2, 2, 6, 2, 6, 6, 2, 6]],
            }
        )

    ann = {
        "images": [{"id": 1, "file_name": "0001.png", "width": 16, "height": 16}],
        "annotations": annotations,
        "categories": [{"id": category_id, "name": f"class_{category_id}"} for category_id in category_ids],
    }
    ann_path = root / "annotations" / "instances_train.json"
    ann_path.write_text(json.dumps(ann) + "\n", encoding="utf-8")
    return ann_path


def test_validate_config_rejects_magformer_num_classes_not_equal_to_one() -> None:
    config = load_config(
        "configs/magformer_aligned_comparison.yaml",
        overrides={
            "data": {"dataset_root": "/tmp/dummy_dataset"},
            "model": {"magformer": {"sem_seg_head": {"num_classes": 2}}},
        },
    )

    assert validate_config(config, strict=False) is False


def test_coco_rgbd_dataset_rejects_multi_class_annotations(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _write_dataset(root, category_ids=[1, 2])

    with pytest.raises(ValueError, match="single-class"):
        CocoRgbdDataset(
            dataset_root=str(root),
            ann_file="annotations/instances_train.json",
            split="train",
            transform=None,
            is_train=True,
        )


def test_evaluate_cli_loads_config_weights_when_flag_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from tools import evaluate as evaluate_tool

    load_calls = []

    args = Namespace(
        config_file="configs/magformer_aligned_comparison.yaml",
        dataset_root=None,
        weights=None,
        output=str(tmp_path / "eval_out"),
        batch_size=1,
        num_workers=0,
    )
    config = SimpleNamespace(
        data=SimpleNamespace(dataset_root=str(tmp_path / "ds")),
        runtime=SimpleNamespace(output_dir=str(tmp_path / "eval_out"), seed=0),
        model=SimpleNamespace(weights="from_config.pth"),
    )
    dataset = SimpleNamespace(coco=object())

    class _Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def forward_inference_raw(self, images, depths):
            del images, depths
            return {"predictions": []}

    monkeypatch.setattr(evaluate_tool, "parse_args", lambda: args)
    monkeypatch.setattr(evaluate_tool, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(evaluate_tool, "setup_device", lambda runtime: torch.device("cpu"))
    monkeypatch.setattr(evaluate_tool, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        evaluate_tool,
        "build_val_loader",
        lambda *a, **k: (
            dataset,
            [{"images": torch.zeros(1, 3, 8, 8), "depths": torch.zeros(1, 1, 8, 8), "image_ids": [1]}],
        ),
    )
    monkeypatch.setattr(evaluate_tool, "build_model", lambda config: _Model())
    monkeypatch.setattr(
        evaluate_tool,
        "load_checkpoint",
        lambda path, model, strict=False: load_calls.append((path, strict)),
    )
    monkeypatch.setattr(
        evaluate_tool,
        "run_inference_evaluation",
        lambda *a, **k: SimpleNamespace(
            coco_results_path=Path(args.output) / "coco_instances_results.json",
            coco_metrics={"segm_AP": 0.5},
        ),
    )

    evaluate_tool.main()

    assert load_calls == [("from_config.pth", False)]


def test_evaluate_cli_requires_a_weight_source(monkeypatch, tmp_path: Path) -> None:
    from tools import evaluate as evaluate_tool

    args = Namespace(
        config_file="configs/magformer_aligned_comparison.yaml",
        dataset_root=None,
        weights=None,
        output=str(tmp_path / "eval_out"),
        batch_size=1,
        num_workers=0,
    )
    config = SimpleNamespace(
        data=SimpleNamespace(dataset_root=str(tmp_path / "ds")),
        runtime=SimpleNamespace(output_dir=str(tmp_path / "eval_out"), seed=0),
        model=SimpleNamespace(weights=None),
    )

    monkeypatch.setattr(evaluate_tool, "parse_args", lambda: args)
    monkeypatch.setattr(evaluate_tool, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(evaluate_tool, "setup_device", lambda runtime: torch.device("cpu"))
    monkeypatch.setattr(evaluate_tool, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        evaluate_tool,
        "build_val_loader",
        lambda *a, **k: (
            SimpleNamespace(coco=object()),
            [{"images": torch.zeros(1, 3, 8, 8), "depths": torch.zeros(1, 1, 8, 8), "image_ids": [1]}],
        ),
    )
    monkeypatch.setattr(evaluate_tool, "build_model", lambda config: object())

    with pytest.raises(ValueError, match="weights"):
        evaluate_tool.main()
