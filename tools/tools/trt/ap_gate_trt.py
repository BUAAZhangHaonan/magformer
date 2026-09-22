#!/usr/bin/env python3
"""AP-gate experiment: towers-TRT vs eager fp32, subset-100 COCO segm AP.

Proper acceptance gate for a compiled engine is AP-within-tolerance (bit
identity is impossible across frameworks by construction). This rebuilds
the towers TRT engine and runs the paired evaluation.
"""
import json
import sys
import time
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "tools/trt"))

CFG = str(SRC / "configs" / "perf_c0_infer_bench.yaml")
WEIGHTS = "/home/hdd3/zhanghaonan/magformer/archive_20260906/staging_4028/home/g203-4028/magformer/output/experiments/next_stage/c0_corrected_300k_seed42/model_best.pth"
KEYS = ("res2", "res3", "res4", "res5")
N = 100


def main():
    from onnx_build_bench import build_engine, TrtRunner
    from magformer.config import load_config
    from magformer.models import build_model
    from magformer.engine.utils import load_checkpoint
    from magformer.data import CocoRgbdDataset
    from magformer.data.transforms import RGBDTransform
    from magformer.data.collate import collate_fn
    from magformer.engine.coco_export import outputs_to_coco_instances
    from magformer.engine.evaluator import COCOEvaluator

    dev = torch.device("cuda:0")
    config = load_config(CFG)
    model = build_model(config)
    load_checkpoint(WEIGHTS, model, strict=False)
    model = model.to(dev).eval()

    d = config.data
    transform = RGBDTransform(image_size=d.image_size, min_scale=d.min_scale, max_scale=d.max_scale,
                              random_flip="none", rgb_brightness=0.0, rgb_contrast=0.0,
                              rgb_saturation=0.0, rgb_hue=0.0, depth_scale=d.depth.scale,
                              depth_shift=d.depth.shift, depth_clip_min=d.depth.clip_min,
                              depth_clip_max=d.depth.clip_max, depth_norm=d.depth.norm,
                              depth_per_sample_norm=False, is_train=False)
    ds = CocoRgbdDataset(dataset_root=d.dataset_root, ann_file="annotations/instances_val.validated.json",
                         split="val", transform=transform, is_train=False)
    cat_ids = list(getattr(ds, "category_ids", [])) or None

    # --- rebuild towers engine ---
    class Towers(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.rgb = m.rgb_backbone
            self.depth = m.depth_backbone
            self.register_buffer("pm", m.pixel_mean.detach().clone())
            self.register_buffer("ps", m.pixel_std.detach().clone())

        def forward(self, images, depths):
            inorm = (images - self.pm.view(1, -1, 1, 1)) / self.ps.view(1, -1, 1, 1)
            r = self.rgb(inorm)
            dd = self.depth(depths)
            return (*(r[k] for k in KEYS), *(dd[k] for k in KEYS))

    towers = Towers(model).eval()
    batch0 = collate_fn([ds[0]])
    im0 = batch0["images"].to(dev); dp0 = batch0["depths"].to(dev)
    inner = model.rgb_backbone.model
    inner._tracing_fixed_hw = (256, 256)
    onnx_path = "/tmp/c0_towers.onnx"
    if not Path(onnx_path).exists():
        with torch.no_grad():
            torch.onnx.export(towers, (im0, dp0), onnx_path,
                              input_names=["images", "depths"],
                              output_names=[f"rgb_{k}" for k in KEYS] + [f"dep_{k}" for k in KEYS],
                              opset_version=17, do_constant_folding=True)
    import os
    half = os.environ.get("TRT_FP16", "0") == "1"
    engine_path = str(SRC / "tools/trt" / ("c0_towers_fp16.engine" if half else "c0_towers_fp32.engine"))
    if not Path(engine_path).exists():
        build_engine(onnx_path, engine_path, fp16=half, workspace_gb=6)
    runner = TrtRunner(engine_path)
    in_dtype = torch.half if half else torch.float32

    orig = model._extract_dual_features

    def patched(images, depths, depth_valid_masks=None, depth_noise_masks=None):
        outs = runner(images.to(in_dtype), depths.to(in_dtype))
        rgb_features = {k: outs[f"rgb_{k}"].float() for k in KEYS}
        depth_features = {k: outs[f"dep_{k}"].float() for k in KEYS}
        images_norm = (images - model.pixel_mean.view(1, -1, 1, 1)) / model.pixel_std.view(1, -1, 1, 1)
        fused, conf, _l = model.fusion(
            image_features=rgb_features, depth_features=depth_features,
            depth_raw=depths, rgb_image=images_norm, depth_noise_mask=depth_noise_masks)
        return fused, conf, _l or {}

    def run_eval(tag):
        ev = COCOEvaluator(coco_gt=ds.coco, iou_types=["bbox", "segm"], max_dets=100)
        t0 = time.time()
        for i in range(N):
            b = collate_fn([ds[i]])
            im = b["images"].to(dev); dp = b["depths"].to(dev)
            pad = b.get("padding_masks"); pad = pad.to(dev) if pad is not None else torch.zeros(im.shape[0], *im.shape[-2:], dtype=torch.bool, device=dev)
            dvm = b.get("depth_valid_masks"); dvm = dvm.to(dev) if dvm is not None else torch.ones_like(dp, dtype=torch.bool)
            dnm = b.get("noise_masks"); dnm = dnm.to(dev) if dnm is not None else torch.zeros_like(dp, dtype=torch.bool)
            with torch.inference_mode():
                outputs = model.forward_inference_raw(im, dp, padding_masks=pad, depth_valid_masks=dvm,
                                                      depth_noise_masks=dnm, gpu_export=True)
            preds = outputs_to_coco_instances(outputs=outputs, image_ids=b["image_ids"],
                                              score_threshold=0.0, mask_threshold=0.5,
                                              category_offset=1, category_ids=cat_ids)
            ev.update(preds, image_ids=b["image_ids"])
        m = ev.summarize()
        def get_seg(md):
            if "segm_AP" in md: return md["segm_AP"]
            for v in md.values():
                if isinstance(v, dict) and "segm_AP" in v: return v["segm_AP"]
            return None
        seg = get_seg(m)
        print(f"[{tag}] subset-{N} segm_AP = {seg}  ({time.time()-t0:.1f}s)")
        return seg

    seg_trt = run_eval("towers-TRT" + ("-fp16" if half else "-fp32"))
    model._extract_dual_features = orig
    seg_eager = run_eval("eager   ")
    print(json.dumps({"segm_ap_eager": seg_eager, "segm_ap_towers_trt": seg_trt,
                      "delta_pt": round((seg_trt - seg_eager) * 100, 3)}, indent=1))


if __name__ == "__main__":
    main()
