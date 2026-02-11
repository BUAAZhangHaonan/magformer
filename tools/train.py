#!/usr/bin/env python3
"""
MAGFormer Training Script

This script trains the MAGFormer model on RGB-D instance segmentation datasets.

Usage:
    python train.py --config configs/mgm_swin_convnext.yaml --num-gpus 4
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from detectron2.engine import launch
from mask2former.train_net import Trainer


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train MAGFormer model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--num-machines", type=int, default=1,
                        help="Number of machines for distributed training")
    parser.add_argument("--machine-rank", type=int, default=0,
                        help="Machine rank for distributed training")
    parser.add_argument("--port", type=int, default=29500,
                        help="Port for distributed training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    return parser.parse_args()


def main(args):
    """Main training function."""
    import detectron2.utils.comm as comm
    from detectron2.config import get_cfg
    from detectron2.engine import default_setup, default_argument_parser, launch
    from detectron2.utils.logger import setup_logger

    from mask2former import add_mgm_config
    import mask2former.data.datasets.register_coco_rgbd_instance

    # Setup logger
    logger = setup_logger()

    # Load config
    cfg = get_cfg()
    add_mgm_config(cfg)
    cfg.merge_from_file(args.config)

    # Override settings
    if args.resume:
        cfg.resume_or_load = True

    # Setup
    default_setup(cfg, args)

    # Create trainer
    trainer = Trainer(cfg)

    # Evaluation only
    if args.eval_only:
        trainer.resume_or_load(resume=args.resume)
        return trainer.test(cfg, trainer.model)

    # Training
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = parse_args()

    # Launch training
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=f"tcp://127.0.0.1:{args.port}",
        args=(args,),
    )
