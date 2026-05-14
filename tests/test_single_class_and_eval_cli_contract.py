from __future__ import annotations

import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _ensure_pycocotools_stub() -> None:
    if "pycocotools" in sys.modules:
        return

    pycocotools = types.ModuleType("pycocotools")
    coco_mod = types.ModuleType("pycocotools.coco")
    cocoeval_mod = types.ModuleType("pycocotools.cocoeval")
    mask_mod = types.ModuleType("pycocotools.mask")

    class COCO:
        def __init__(self, ann_file=None):
            self.ann_file = ann_file
            self.dataset = {}
            self.imgs = {}
            if ann_file is not None:
                with open(ann_file, "r", encoding="utf-8") as handle:
                    self.dataset = json.load(handle)
                self.imgs = {
                    int(image["id"]): image
                    for image in self.dataset.get("images", [])
                    if "id" in image
                }

        def getAnnIds(self, imgIds=None):
            ann_ids = []
            for ann in self.dataset.get("annotations", []):
                if imgIds is None or ann.get("image_id") == imgIds:
                    ann_ids.append(int(ann.get("id", len(ann_ids) + 1)))
            return ann_ids

        def loadImgs(self, img_ids):
            if not isinstance(img_ids, (list, tuple)):
                img_ids = [img_ids]
            return [self.imgs[int(img_id)] for img_id in img_ids]

        def loadAnns(self, ann_ids):
            annotations = self.dataset.get("annotations", [])
            ann_id_set = {int(ann_id) for ann_id in ann_ids}
            return [ann for ann in annotations if int(ann.get("id", -1)) in ann_id_set]

    def encode(array):
        ys, xs = np.where(array > 0)
        bbox = [0.0, 0.0, 0.0, 0.0]
        if len(xs) > 0 and len(ys) > 0:
            bbox = [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() - xs.min() + 1),
                float(ys.max() - ys.min() + 1),
            ]
        return {"size": list(array.shape), "counts": b"1", "bbox": bbox}

    def toBbox(rle):
        return np.asarray(rle.get("bbox", [0.0, 0.0, 0.0, 0.0]), dtype=float)

    coco_mod.COCO = COCO
    cocoeval_mod.COCOeval = object
    mask_mod.encode = encode
    mask_mod.toBbox = toBbox
    pycocotools.coco = coco_mod
    pycocotools.cocoeval = cocoeval_mod
    pycocotools.mask = mask_mod
    pycocotools.__path__ = []  # type: ignore[attr-defined]
    sys.modules["pycocotools"] = pycocotools
    sys.modules["pycocotools.coco"] = coco_mod
    sys.modules["pycocotools.cocoeval"] = cocoeval_mod
    sys.modules["pycocotools.mask"] = mask_mod


def _ensure_cv2_stub() -> None:
    if "cv2" in sys.modules:
        return

    cv2_mod = types.ModuleType("cv2")
    cv2_mod.imwrite = lambda path, image: True
    sys.modules["cv2"] = cv2_mod


_ensure_pycocotools_stub()
_ensure_cv2_stub()

from magformer.config import load_config
from magformer.config.validation import validate_config
from magformer.data import CocoRgbdDataset


def _write_dataset(root: Path, *, category_ids: list[int]) -> Path:
    (root / "images" / "train").mkdir(parents=True)
    (root / "depth" / "depth_npy" / "train").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    (root / "images" / "train" / "0001.png").write_bytes(b"placeholder-image")
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
        "configs/base.yaml",
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



class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.transform = None

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {
            "images": torch.zeros(3, 8, 8),
            "depths": torch.zeros(1, 8, 8),
            "image_ids": index,
        }


def test_runtime_config_accepts_eval_batch_size() -> None:
    from magformer.config.schema import RuntimeConfig

    cfg = RuntimeConfig(eval_batch_size=4)

    assert cfg.eval_batch_size == 4


def test_train_val_loader_uses_runtime_eval_batch_size(monkeypatch) -> None:
    from tools import train as train_tool

    monkeypatch.setattr(train_tool, "is_vc_suda_enabled", lambda config: False)
    monkeypatch.setattr(
        "magformer.data.transforms.RGBDTransform",
        lambda *args, **kwargs: ("transform", args, kwargs),
    )
    config = SimpleNamespace(
        data=SimpleNamespace(
            image_size=8,
            min_scale=1.0,
            max_scale=1.0,
            random_flip="none",
            rgb_photo_aug=SimpleNamespace(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0),
            depth=SimpleNamespace(scale=1.0, shift=0.0, clip_min=0.0, clip_max=1.0, norm="minmax", per_sample_norm=True),
            depth_noise=SimpleNamespace(gaussian_std=0.0, speckle_std=0.0, drop_prob=0.0, drop_val=0.0),
        ),
        runtime=SimpleNamespace(eval_batch_size=4),
    )

    _, val_loader = train_tool.build_data_loaders(
        config,
        _TinyDataset(),
        _TinyDataset(),
        batch_size=2,
        num_workers=0,
        is_distributed=False,
    )

    assert val_loader.batch_size == 4

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
    eval_kwargs = []

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
        lambda *a, **k: eval_kwargs.append(k) or SimpleNamespace(
            coco_results_path=Path(args.output) / "coco_instances_results.json",
            coco_metrics={"segm_AP": 0.5},
        ),
    )
    config.runtime.eval_iou_types = ["bbox", "segm"]
    config.runtime.eval_max_images = 28

    evaluate_tool.main()

    assert load_calls == [("from_config.pth", True)]
    assert eval_kwargs[-1]["iou_types"] == ["bbox", "segm"]
    assert eval_kwargs[-1]["max_images"] == 28
    assert eval_kwargs[-1]["fail_on_empty"] is True


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


def test_inference_write_visualization_fails_loudly(monkeypatch, tmp_path: Path) -> None:
    from tools import inference as inference_tool

    monkeypatch.setattr(inference_tool.cv2, "COLOR_RGB2BGR", 1, raising=False)
    monkeypatch.setattr(inference_tool.cv2, "cvtColor", lambda image, code: image, raising=False)
    monkeypatch.setattr(inference_tool.cv2, "imwrite", lambda path, image: False, raising=False)

    with pytest.raises(RuntimeError, match="Failed to write visualization"):
        inference_tool.write_visualization(
            tmp_path / "missing" / "out.png",
            np.zeros((4, 4, 3), dtype=np.uint8),
        )
