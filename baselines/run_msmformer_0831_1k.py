#!/usr/bin/env python3
"""
Run MSMFormer (ICRA 2026 baseline) on ECC 0831_1K with COCOeval (segm/bbox AP).

This is a thin wrapper that:
- registers the ECC RGBD COCO dataset in-process (adds `depth_file_name`)
- wires a minimal RGBD DatasetMapper (reads `.npy` depth)
- runs Detectron2 training/eval with MSMFormer code vendored under:
  `baselines/icra_2026/msmformer/`
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as d2_utils
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.utils.logger import setup_logger

# Ensure sibling baseline utilities are importable when running as a file.
BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from register_0831_1k_coco_rgbd import register_0831_1k_coco_rgbd
from depth_stats import load_0831_1k_depth_stats


def _split_args(argv: List[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


class DatasetMapperRGBD:
    """
    Minimal Detectron2 dataset mapper that provides:
    - image: float32 tensor (3,H,W), normalized using cfg.MODEL.PIXEL_MEAN/STD
    - depth: float32 tensor (3,H,W), loaded from `.npy` and resized/flipped identically
    """

    def __init__(self, cfg, is_train: bool):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.pixel_mean = np.asarray(cfg.MODEL.PIXEL_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.pixel_std = np.asarray(cfg.MODEL.PIXEL_STD, dtype=np.float32).reshape(1, 1, 3)

        stats = load_0831_1k_depth_stats()
        self.depth_min = float(stats.p1)
        self.depth_max = float(stats.p99)

        # Fixed-size baseline protocol: resize to 512x512; apply the same transform to RGB+depth.
        target = (int(cfg.INPUT.MIN_SIZE_TEST), int(cfg.INPUT.MIN_SIZE_TEST))
        augs: List[T.Augmentation] = [T.Resize(target)]
        if is_train:
            augs.insert(0, T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
        self.augmentations = T.AugmentationList(augs)

    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        dataset_dict = copy.deepcopy(dataset_dict)

        image = d2_utils.read_image(dataset_dict["file_name"], format=self.image_format)
        depth_path = dataset_dict.get("depth_file_name")
        if not depth_path:
            raise KeyError("Missing `depth_file_name` in dataset dict (did you register RGBD dataset?)")

        depth = np.load(depth_path).astype(np.float32)
        if depth.ndim == 2:
            depth = depth[:, :, None]
        if depth.shape[2] == 1:
            depth = np.repeat(depth, 3, axis=2)

        if depth.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Depth shape {depth.shape[:2]} does not match RGB shape {image.shape[:2]} for: {dataset_dict['file_name']}"
            )

        combo = np.concatenate([image.astype(np.float32), depth], axis=2)  # (H,W,6)
        aug_input = T.AugInput(combo)
        transforms = self.augmentations(aug_input)
        combo = aug_input.image

        image = combo[:, :, :3]
        depth = combo[:, :, 3:6]
        image_shape = image.shape[:2]  # (H,W)

        # Global depth normalization to [0,1] using ECC 0831_1K train stats.
        depth = np.clip(depth, self.depth_min, self.depth_max)
        depth = (depth - self.depth_min) / (self.depth_max - self.depth_min + 1e-6)

        # Normalize RGB only. Depth stays float32 (0..1-ish) and is consumed by the UCN-style backbone.
        image = (image - self.pixel_mean) / self.pixel_std

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float32)
        dataset_dict["depth"] = torch.as_tensor(np.ascontiguousarray(depth.transpose(2, 0, 1)), dtype=torch.float32)

        if "annotations" in dataset_dict:
            annos = [
                d2_utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = d2_utils.annotations_to_instances(annos, image_shape, mask_format=self.mask_format)
            dataset_dict["instances"] = d2_utils.filter_empty_instances(instances)

        return dataset_dict


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapperRGBD(cfg, is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapperRGBD(cfg, is_train=False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=True, output_dir=output_folder)


def setup(args) -> Any:
    cfg = get_cfg()

    # MSMFormer config extension.
    # NOTE: The vendored code path is injected in `main()` before calling setup().
    from meanshiftformer.config import add_meanshiftformer_config

    add_meanshiftformer_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=0, name="msmformer")
    return cfg


def _write_trainable_params(model: torch.nn.Module, out_dir: str) -> None:
    p = Path(out_dir) / "params_trainable.txt"
    n = sum(int(x.numel()) for x in model.parameters() if x.requires_grad)
    p.write_text(str(n) + "\n", encoding="utf-8")


def main(args, dataset_root: str, msmformer_root: str) -> Dict[str, Any] | None:
    # Register datasets in-process (RGBD).
    register_0831_1k_coco_rgbd(dataset_root)

    # MSMFormer uses an internal UCN-style backbone configured via `lib/fcn/config.py`.
    # Enforce scratch-only policy + RGBD mode for ECC baselines.
    from fcn.config import cfg as ucn_cfg

    ucn_cfg.INPUT = "RGBD"
    ucn_cfg.TRAIN.FUSION_TYPE = "add"
    ucn_cfg.TRAIN.EMBEDDING_PRETRAIN = False

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if args.eval_only and args.config_file:
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    _write_trainable_params(trainer.model, cfg.OUTPUT_DIR)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def cli() -> None:
    wrapper_argv, passthrough = _split_args(sys.argv[1:])

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to magformer_datasets/0831_1K (default: workspace-relative).",
    )
    ap.add_argument(
        "--msmformer-root",
        type=str,
        default="baselines/icra_2026/msmformer/MSMFormer",
        help="Path to vendored MSMFormer code root (contains `meanshiftformer/`).",
    )
    argsw = ap.parse_args(wrapper_argv)

    # Default dataset root is workspace-relative: <ws>/magformer_datasets/0831_1K
    if argsw.dataset_root is None:
        ws_root = Path(__file__).resolve().parents[2]
        argsw.dataset_root = str(ws_root / "magformer_datasets" / "0831_1K")

    msm_root = Path(argsw.msmformer_root).resolve()
    if not msm_root.exists():
        raise FileNotFoundError(f"msmformer root not found: {msm_root}")

    # Make `import meanshiftformer` work and ensure UCN-style `lib/` is discoverable.
    sys.path.insert(0, str(msm_root))
    sys.path.insert(0, str(msm_root.parent))
    sys.path.insert(0, str(msm_root.parent / "tools"))

    # Populate Detectron2 registries (meta-arch, heads, etc.).
    import meanshiftformer  # noqa: F401

    args = default_argument_parser().parse_args(passthrough)
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, argsw.dataset_root, str(msm_root)),
    )


if __name__ == "__main__":
    cli()
