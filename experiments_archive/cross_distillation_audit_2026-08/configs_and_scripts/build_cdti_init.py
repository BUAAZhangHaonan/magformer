"""Build cdti_init.pth = coco4ch_magformer_init.pth + distilled depth tower.

FiLM keys are deliberately omitted: the constructor zero-inits them, and the
warm-start loader tolerates missing keys, so they stay exactly zero at load.
"""
import sys

import torch

BASE = "/home/hdd3/zhanghaonan/magformer_audit/pretrained/coco4ch_magformer_init.pth"
TOWER = "/home/hdd3/zhanghaonan/magformer_audit/pretrained/depth_tower_distilled.pth"
OUT = "/home/hdd3/zhanghaonan/magformer_audit/pretrained/cdti_init.pth"

base = torch.load(BASE, map_location="cpu", weights_only=False)
sd = base["model_state_dict"] if "model_state_dict" in base else base["model"]

tower = torch.load(TOWER, map_location="cpu", weights_only=False)["tower"]

added = {}
for k, v in tower.items():
    added["depth_tower." + k] = v

CHANS = (16, 32, 64, 128)
DIMS = (96, 192, 384, 768)
for i, (c, d) in enumerate(zip(CHANS, DIMS)):
    added[f"depth_films.{i}.conv.weight"] = torch.zeros(2 * d, c, 1, 1)
    added[f"depth_films.{i}.conv.bias"] = torch.zeros(2 * d)

sd.update(added)
out = {"model_state_dict": sd}
torch.save(out, OUT)
print(f"base keys={len(base['model_state_dict']) if 'model_state_dict' in base else len(base)} tower_added={len(added)} total={len(sd)}")
print("saved", OUT)
