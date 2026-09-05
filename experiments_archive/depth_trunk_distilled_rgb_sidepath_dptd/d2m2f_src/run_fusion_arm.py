"""E24 fusion arms: concat4ch (anchor) vs SGF vs hybrid, M2F Swin-T, 1K RGBD dataset.

Usage: python run_fusion_arm.py --arm concat4ch|sgf|hybrid [--lrboost] [--smoke] [--debug-film]
"""
import argparse
import sys

SRC = "/home/hdd1/wanghaoran/magformer/d2m2f_src"
sys.path.insert(0, SRC)
sys.path.insert(0, f"{SRC}/Mask2Former")

import register_32254_rgbd  # noqa: F401
import concat4ch_backbone  # noqa: F401  registers D2SwinTransformer4CH
import sgf_backbone  # noqa: F401  registers D2SwinTransformerSGF
import hybrid_backbone  # noqa: F401  registers D2SwinTransformerHybrid
import mgc_backbone  # noqa: F401  registers D2SwinTransformerMGC
import modality_backbones  # noqa: F401  registers RGBOnly/DepthOnly
import dptd_backbone  # noqa: F401  registers D2SwinTransformerDPTD
from rgbd_concat_mapper import RGBDConcatMapper

from detectron2.data import build_detection_test_loader, build_detection_train_loader
from train_net import Trainer, main

M2F_CFG = f"{SRC}/Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml"
P4CH = "/home/hdd1/wanghaoran/magformer/pretrained/m2f_4ch_zero_clsadapt.pth"
P3CH = "/home/hdd1/wanghaoran/magformer/pretrained/m2f_3ch_clsadapt.pth"
OUT_ROOT = "/home/hdd1/wanghaoran/magformer/output/experiments/E24-fusion-6401-20260821"
_P4CH_ARMS = {"concat4ch", "hybrid", "hybrid_ptd", "mgc", "mgc2", "concat4ch_caug", "mgc2_caug"}  # 4ch-stem arms

parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True, choices=["concat4ch", "sgf", "hybrid", "mgc", "mgc2",
                                                     "concat4ch_caug", "mgc2_caug",
                                                     "rgb_only", "depth_only", "hybrid_ptd", "dptd"])
parser.add_argument("--lrboost", action="store_true",
                    help="give depth_tower/films params full LR (undo backbone x0.1)")
parser.add_argument("--weights", default=None, help="override MODEL.WEIGHTS (warm start)")
parser.add_argument("--base-lr", default=None)
parser.add_argument("--stemboost", action="store_true",
                    help="patch_embed params at 10x LR (fresh stem under warm trunk)")
parser.add_argument("--backbone-mult", default=None)
parser.add_argument("--out-suffix", default="", help="append to OUTPUT_DIR (distinct pilot dirs)")
parser.add_argument("--steps", default=None, help="override SOLVER.STEPS, e.g. (45000,)")
parser.add_argument("--tower-init", default=None,
                    help="path to a distilled DepthTower state_dict loaded into the backbone at build")
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--debug-film", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--max-iter", type=int, default=10000)
parser.add_argument("--eval-period", type=int, default=None)
args = parser.parse_args()

# patch Trainer loaders to use the RGBD mapper (train + eval both need 4ch)
def _build_train_loader(cls, cfg):
    return build_detection_train_loader(cfg, mapper=RGBDConcatMapper(cfg, is_train=True))

def _build_test_loader(cls, cfg, dataset_name):
    return build_detection_test_loader(
        cfg, dataset_name, mapper=RGBDConcatMapper(cfg, is_train=False)
    )

Trainer.build_train_loader = classmethod(_build_train_loader)
Trainer.build_test_loader = classmethod(_build_test_loader)

if args.lrboost:
    _orig_bo = Trainer.build_optimizer  # M2F's own override (per-param groups, full_model clip)

    def _build_optimizer(cls, cfg, model):
        opt = _orig_bo(cfg, model)
        boost_ids = set()
        for name, mod in model.backbone.named_modules():
            if "depth_tower" in name or "rgb_tower" in name or "films" in name or "depth_gate" in name:
                for p in mod.parameters(recurse=True):
                    boost_ids.add(id(p))
        stem_ids = set()
        if args.stemboost:
            for p in model.backbone.patch_embed.parameters():
                stem_ids.add(id(p))
        n = 0
        for pg in opt.param_groups:
            for p in pg["params"]:
                if id(p) in boost_ids:
                    pg["lr"] *= 10.0  # undo BACKBONE_MULTIPLIER 0.1 (per-param groups)
                    n += 1
                elif id(p) in stem_ids:
                    pg["lr"] *= 10.0
                    n += 1
        print(f"[lrboost] {n} depth-module params -> full LR", flush=True)
        return opt

    Trainer.build_optimizer = classmethod(_build_optimizer)

max_iter = 60 if args.smoke else args.max_iter
ckpt_period = 60 if args.smoke else 2500
eval_period = ckpt_period if args.eval_period is None else args.eval_period

sys.argv = [
    "train_net.py",
    "--config-file", M2F_CFG,
    "MODEL.WEIGHTS", args.weights or (P4CH if args.arm in _P4CH_ARMS else P3CH),
    "MODEL.BACKBONE.NAME",
    {"concat4ch": "D2SwinTransformer4CH", "sgf": "D2SwinTransformerSGF",
     "hybrid": "D2SwinTransformerHybrid", "mgc": "D2SwinTransformerMGC",
     "mgc2": "D2SwinTransformerMGC", "concat4ch_caug": "D2SwinTransformer4CH",
     "mgc2_caug": "D2SwinTransformerMGC", "rgb_only": "D2SwinTransformerRGBOnly",
     "depth_only": "D2SwinTransformerDepthOnly",
     "hybrid_ptd": "D2SwinTransformerHybrid", "dptd": "D2SwinTransformerDPTD"}[args.arm],
    "MODEL.PIXEL_MEAN", "[123.675,116.28,103.53,125.628]",
    "MODEL.PIXEL_STD", "[58.395,57.12,57.375,15.157]",
    "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "1",
    "DATASETS.TRAIN", "(\"ecc20260318_1k_rgbd_train\",)",
    "DATASETS.TEST", "(\"ecc20260318_1k_rgbd_val_subset1000\",)",
    "TEST.EVAL_PERIOD", str(eval_period),
    "DATALOADER.NUM_WORKERS", "2",
    "SOLVER.IMS_PER_BATCH", "2",
    "SOLVER.BASE_LR", args.base_lr or "5e-5",
    "SOLVER.WEIGHT_DECAY", "0.05",
    "SOLVER.STEPS", args.steps or "(999999,)",
    "SOLVER.WARMUP_ITERS", "10",
    "SOLVER.MAX_ITER", str(max_iter),
    "SOLVER.CHECKPOINT_PERIOD", str(ckpt_period),
    "SOLVER.AMP.ENABLED", "True",
    "SOLVER.CLIP_GRADIENTS.ENABLED", "True",
    "SOLVER.CLIP_GRADIENTS.CLIP_TYPE", "full_model",
    "SOLVER.CLIP_GRADIENTS.CLIP_VALUE", "0.01",
    "SOLVER.BACKBONE_MULTIPLIER", args.backbone_mult or "0.1",
    "INPUT.MASK_FORMAT", "bitmask",
    "SEED", "42",
    "OUTPUT_DIR", f"{OUT_ROOT}/{args.arm}" + ("_lrboost" if args.lrboost else "")
                  + args.out_suffix + ("_smoke" if args.smoke else ""),
]
print(f"[output-dir] {sys.argv[sys.argv.index('OUTPUT_DIR') + 1]}", flush=True)

a = type("A", (), {})()
a.config_file = sys.argv[2]
a.num_gpus = 1
a.num_machines = 1
a.machine_rank = 0
a.dist_url = "auto"
a.eval_only = False
a.resume = args.resume
a.opts = sys.argv[3:]

if args.tower_init:
    import torch as _torch
    _orig_build_model = Trainer.build_model

    def _build_model_tower(cls, cfg):
        model = _orig_build_model(cfg)
        sd = _torch.load(args.tower_init, map_location="cpu", weights_only=False)["tower"]
        tgt = model.backbone.rgb_tower if hasattr(model.backbone, "rgb_tower") else model.backbone.depth_tower
        tgt.load_state_dict(sd)
        print(f"[tower-init] loaded {len(sd)} tensors from {args.tower_init}", flush=True)
        return model

    Trainer.build_model = classmethod(_build_model_tower)

if args.debug_film and args.arm in ("sgf", "hybrid"):
    _orig_build_model = Trainer.build_model

    def _build_model_dbg(cls, cfg):
        model = _orig_build_model(cfg)
        model.backbone._debug_film = True
        return model

    Trainer.build_model = classmethod(_build_model_dbg)

main(a)
