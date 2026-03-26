#!/usr/bin/env python3
"""
Run MSMFormer (ICRA 2026 baseline) on ECC 0831_1K with COCOeval (segm/bbox AP).

This is a thin wrapper that:
- registers the ECC RGBD COCO dataset in-process (adds `depth_file_name`)
- wires a minimal RGBD DatasetMapper (reads `.npy` depth)
- runs Detectron2 training/eval with MSMFormer code vendored under:
  `baselines/msmformer/`
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as d2_utils
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.utils.logger import setup_logger

# Ensure sibling baseline utilities are importable when running as a file.
BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from register_0831_1k_coco_rgbd import register_0831_1k_coco_rgbd
from depth_stats import load_0831_1k_depth_stats
from rgbd_geometry import depth_to_xyz
from run_msmformer_ecc import (
    _ensure_msmformer_import_paths,
    _find_model_weights_opt,
    _with_default_model_weights,
    build_msmformer_optimizer,
    build_msmformer_recipe,
)


def _split_args(argv: List[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


class DatasetMapperRGBD:
    """
    Minimal Detectron2 dataset mapper that provides:
    - image: float32 tensor (3,H,W), normalized according to the active backbone path
    - depth: float32 tensor (3,H,W), loaded from `.npy` and resized/flipped identically
    """

    def __init__(self, cfg, is_train: bool):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.use_other_backbone = bool(cfg.MODEL.USE_OTHER_BACKBONE)
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
        depth_hw = depth[:, :, 0] if depth.ndim == 3 and depth.shape[2] == 1 else depth
        if depth_hw.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Depth shape {depth_hw.shape[:2]} does not match RGB shape {image.shape[:2]} for: {dataset_dict['file_name']}"
            )

        if depth_hw.ndim != 2:
            raise ValueError(f"Expected scalar depth map from dataset, got shape={depth_hw.shape}")

        combo = np.concatenate([image.astype(np.float32), depth_hw[:, :, None]], axis=2)  # (H,W,4)
        aug_input = T.AugInput(combo)
        transforms = self.augmentations(aug_input)
        combo = aug_input.image

        image = combo[:, :, :3]
        depth = combo[:, :, 3]
        image_shape = image.shape[:2]  # (H,W)

        depth = np.clip(depth, self.depth_min, self.depth_max)
        depth = (depth - self.depth_min) / (self.depth_max - self.depth_min + 1e-6)
        depth = depth_to_xyz(depth)

        # The UCN backbone expects mean subtraction in pixel space followed by /255.
        # Other backbones keep standard Detectron2 mean/std normalization.
        if self.use_other_backbone:
            image = (image - self.pixel_mean) / self.pixel_std
        else:
            image = (image - self.pixel_mean) / 255.0

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float32)
        dataset_dict["depth"] = torch.as_tensor(np.ascontiguousarray(depth.transpose(2, 0, 1)), dtype=torch.float32)

        if "annotations" in dataset_dict:
            annos = [
                d2_utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = d2_utils.annotations_to_instances(annos, image_shape, mask_format=self.mask_format)
            instances = d2_utils.filter_empty_instances(instances)
            # MSMFormer target preparation expects dense masks as a tensor (N,H,W).
            # Convert after filtering to keep Detectron2's `nonempty()` checks working.
            if hasattr(instances, "gt_masks"):
                mask_tensor = instances.gt_masks.tensor if hasattr(instances.gt_masks, "tensor") else instances.gt_masks
                label_map = torch.full(image_shape, -1, dtype=torch.int64)
                for inst_id, mask in enumerate(mask_tensor):
                    label_map[mask > 0] = int(inst_id)
                dataset_dict["label"] = label_map.unsqueeze(0)
                if hasattr(instances.gt_masks, "tensor"):
                    instances.gt_masks = mask_tensor
            dataset_dict["instances"] = instances

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

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_msmformer_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)


def setup(args) -> Any:
    cfg = get_cfg()

    # MSMFormer config extension.
    # NOTE: The vendored code path is injected in `main()` before calling setup().
    from meanshiftformer.config import add_meanshiftformer_config

    add_deeplab_config(cfg)
    add_meanshiftformer_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    recipe = build_msmformer_recipe("0831")
    cfg.MODEL.USE_DEPTH = recipe.use_depth
    cfg.MODEL.USE_OTHER_BACKBONE = recipe.use_other_backbone
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = recipe.num_classes
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = recipe.convs_dim
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = recipe.mask_dim
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = recipe.pixel_decoder_name
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = recipe.transformer_in_feature
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = recipe.transformer_decoder_name
    cfg.MODEL.MASK_FORMER.USE_MEANSHIFT_CROSS_ATTENTION = recipe.use_meanshift_cross_attention
    cfg.MODEL.MASK_FORMER.USE_MEANSHIFT_SELF_ATTENTION = recipe.use_meanshift_self_attention
    cfg.MODEL.MASK_FORMER.DISABLE_MEANSHIFT_ATTENTION_MASK = recipe.disable_attention_mask
    cfg.MODEL.MASK_FORMER.DECODER_BLOCK_NORM = recipe.decoder_block_norm
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = recipe.class_weight
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = recipe.mask_weight
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = recipe.dice_weight
    cfg.MODEL.MASK_FORMER.DROPOUT = recipe.dropout
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = recipe.dec_layers
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = recipe.object_mask_threshold
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = recipe.overlap_threshold
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.01
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
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
    recipe = build_msmformer_recipe("0831")

    ucn_cfg.INPUT = recipe.ucn_input_type
    ucn_cfg.TRAIN.FUSION_TYPE = recipe.ucn_fusion_type
    ucn_cfg.TRAIN.EMBEDDING_PRETRAIN = recipe.ucn_embedding_pretrain
    ucn_cfg.TRAIN.EMBEDDING_METRIC = recipe.ucn_embedding_metric
    ucn_cfg.TRAIN.EMBEDDING_NORMALIZATION = recipe.ucn_embedding_normalization
    ucn_cfg.TRAIN.EMBEDDING_LAMBDA_INTRA = recipe.ucn_embedding_lambda_intra
    ucn_cfg.TRAIN.EMBEDDING_LAMBDA_INTER = recipe.ucn_embedding_lambda_inter

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
        default="baselines/msmformer/MSMFormer",
        help="Path to vendored MSMFormer code root (contains `meanshiftformer/`).",
    )
    ap.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help=(
            "Path to official MSMFormer pretrained weights. "
            "Defaults to output/pretrained/norm_RGBD_pretrained.pth when present. "
            "Use 'none' to force scratch."
        ),
    )
    argsw = ap.parse_args(wrapper_argv)

    # Default dataset root is workspace-relative: <ws>/magformer_datasets/0831_1K
    if argsw.dataset_root is None:
        ws_root = Path(__file__).resolve().parents[2]
        argsw.dataset_root = str(ws_root / "magformer_datasets" / "0831_1K")

    msm_root = _ensure_msmformer_import_paths(argsw.msmformer_root)
    if not msm_root.exists():
        raise FileNotFoundError(f"msmformer root not found: {msm_root}")

    # Populate Detectron2 registries (meta-arch, heads, etc.).
    import meanshiftformer  # noqa: F401

    args = default_argument_parser().parse_args(passthrough)
    args.opts = _with_default_model_weights(args.opts, explicit_pretrained=argsw.pretrained)
    resolved_weights = _find_model_weights_opt(args.opts)
    if resolved_weights is not None:
        print(f"[msmformer] loading pretrained checkpoint: {resolved_weights}")
    else:
        print("[msmformer] no pretrained checkpoint found, training from scratch")
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
