import json
import random

SRC = "/home/hdd1/wanghaoran/magformer/magformer_datasets/20260318_1K_32254/annotations/instances_val.validated.json"
DST = "/home/hdd1/wanghaoran/magformer/magformer_datasets/20260318_1K_32254/annotations/instances_val_subset1000.json"

with open(SRC) as f:
    d = json.load(f)

rng = random.Random(42)
imgs = sorted(d["images"], key=lambda x: x["id"])
sel = rng.sample(imgs, 1000)
ids = {im["id"] for im in sel}
anns = [a for a in d["annotations"] if a["image_id"] in ids]

out = dict(d)
out["images"] = sel
out["annotations"] = anns
with open(DST, "w") as f:
    json.dump(out, f)
print(f"subset: {len(sel)} images, {len(anns)} annotations (from {len(d['images'])} / {len(d['annotations'])})")
