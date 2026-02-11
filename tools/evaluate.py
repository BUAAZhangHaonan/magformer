#!/usr/bin/env python3
"""
MAGFormer Evaluation Script

This script evaluates trained MAGFormer models on a test dataset.

Usage:
    python evaluate.py --config configs/mgm_swin_convnext.yaml --model-path output/model_final.pth
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from detectron2.config import get_cfg
from detectron2.engine import default_setup, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger

from mask2former import add_mgm_config
from mask2former.train_net import Trainer
import mask2former.data.datasets.register_coco_rgbd_instance


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate MAGFormer model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name to evaluate on")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results")
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup logger
    logger = setup_logger()

    # Load config
    cfg = get_cfg()
    add_mgm_config(cfg)
    cfg.merge_from_file(args.config)

    # Set model weights
    cfg.MODEL.WEIGHTS = args.model_path

    if args.output:
        cfg.OUTPUT_DIR = args.output

    # Setup
    default_setup(cfg, args)

    # Create trainer (for model building)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)

    # Get dataset name
    if args.dataset is None:
        # Use test dataset from config
        dataset_name = cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else cfg.DATASETS.TEST
    else:
        dataset_name = args.dataset

    logger.info(f"Evaluating on dataset: {dataset_name}")

    # Build data loader
    data_loader = build_detection_test_loader(cfg, dataset_name)

    # Build evaluator
    evaluator = COCOEvaluator(dataset_name, cfg.OUTPUT_DIR, use_fast_impl=False)

    # Run evaluation
    results = inference_on_dataset(trainer.model, data_loader, evaluator)

    # Print results
    logger.info("Evaluation Results:")
    logger.info(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    main()
