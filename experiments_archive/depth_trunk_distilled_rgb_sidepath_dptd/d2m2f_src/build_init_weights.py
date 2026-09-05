"""Build a 4ch zero-padded patch-embed Swin-T checkpoint from the 86143f COCO pkl.

Reads model_final_86143f_official.pkl, pads backbone.patch_embed.weight
[96,3,4,4] -> [96,4,4,4] (4th slice zeros), saves a .pth DetectionCheckpointer
can load directly. New keys untouched; everything else passed through.
"""
import pickle

import torch

SRC = "/home/hdd1/wanghaoran/magformer/pretrained/model_final_86143f_official.pkl"
DST = "/home/hdd1/wanghaoran/magformer/pretrained/m2f_swin_t_4ch_zero_init.pth"
KEY = "backbone.patch_embed.proj.weight"

with open(SRC, "rb") as f:
    ckpt = pickle.load(f, encoding="latin1")

model = ckpt["model"] if "model" in ckpt else ckpt
w = model[KEY]
assert w.shape == (96, 3, 4, 4), w.shape
new = torch.zeros(96, 4, 4, 4)
new[:, :3] = torch.as_tensor(w)
model[KEY] = new
assert set(model) == set(pickle.load(open(SRC, "rb"), encoding="latin1")["model"]) | set()

torch.save({"model": model, "__author__": "e24_concat4ch_zero_init", "matching_heuristics": True}, DST)
print(f"saved {DST}; patch_embed.weight {tuple(new.shape)}, 4th col sum={new[:, 3].abs().sum().item()}")
