import sys
SRC = "/home/hdd1/wanghaoran/magformer/d2m2f_src"
sys.path.insert(0, SRC); sys.path.insert(0, f"{SRC}/Mask2Former")
import register_32254_rgbd
import concat4ch_backbone, sgf_backbone, hybrid_backbone, mgc_backbone, modality_backbones, dptd_backbone
from rgbd_concat_mapper import RGBDConcatMapper
from detectron2.data import build_detection_test_loader
from train_net import Trainer, main
import argparse
p = argparse.ArgumentParser(); p.add_argument("--arm", default="concat4ch", choices=["concat4ch", "sgf", "hybrid", "mgc", "mgc2", "concat4ch_caug", "mgc2_caug", "rgb_only", "depth_only", "hybrid_ptd", "dptd"]); p.add_argument("--weights", required=True); p.add_argument("--dataset", default="ecc20260318_1k_rgbd_val"); p.add_argument("--zero-film", action="store_true", help="zero the FiLM scale/shift at inference (diagnostic)"); a = p.parse_args()
if a.zero_film:
    from sgf_backbone import FiLM as _FiLM
    def _zf(self, feat):
        z = self.conv(feat) * 0.0
        return z.chunk(2, dim=1)
    _FiLM.forward = _zf
def _bte(cls, cfg, name):
    return build_detection_test_loader(cfg, name, mapper=RGBDConcatMapper(cfg, is_train=False))
Trainer.build_test_loader = classmethod(_bte)
name = {"concat4ch": "D2SwinTransformer4CH", "sgf": "D2SwinTransformerSGF", "hybrid": "D2SwinTransformerHybrid", "mgc": "D2SwinTransformerMGC", "mgc2": "D2SwinTransformerMGC", "concat4ch_caug": "D2SwinTransformer4CH", "mgc2_caug": "D2SwinTransformerMGC", "rgb_only": "D2SwinTransformerRGBOnly",
    "depth_only": "D2SwinTransformerDepthOnly", "hybrid_ptd": "D2SwinTransformerHybrid", "dptd": "D2SwinTransformerDPTD"}[a.arm]
class A: pass
_o = A()
_o.eval_only = True; _o.resume = False; _o.num_gpus = 1; _o.dist_url = "tcp://127.0.0.1:9761"
_o.config_file = f"{SRC}/Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml"
_o.opts = [
    "MODEL.WEIGHTS", a.weights, "MODEL.BACKBONE.NAME", name,
    "MODEL.PIXEL_MEAN", "[123.675,116.28,103.53,125.628]", "MODEL.PIXEL_STD", "[58.395,57.12,57.375,15.157]",
    "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "1",
    "DATASETS.TRAIN", "(\"ecc20260318_1k_rgbd_train\",)",
    "DATASETS.TEST", f"(\"{a.dataset}\",)",
    "INPUT.MASK_FORMAT", "bitmask",
    "OUTPUT_DIR", "/home/hdd1/wanghaoran/magformer/output/experiments/E24-fusion-6401-20260821/diag_eval"]
main(_o)
