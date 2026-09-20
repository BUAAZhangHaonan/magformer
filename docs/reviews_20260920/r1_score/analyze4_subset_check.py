import contextlib, io, json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
GT="/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/annotations/instances_val.validated.json"
DET="/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/f1_full_design_256k/eval_snapshots/2001_0920/dets.json"
with contextlib.redirect_stdout(io.StringIO()):
    coco=COCO(GT); dets=json.load(open(DET))
gt_by_img={}
for a in coco.dataset["annotations"]: gt_by_img.setdefault(a["image_id"],[]).append(a)
dets_by_img={}
for i,d in enumerate(dets): dets_by_img.setdefault(d["image_id"],[]).append(i)
def rle_of(d):
    s=d["segmentation"]; c=s["counts"]
    return {"size":s["size"],"counts":c.encode() if isinstance(c,str) else c}
def bucket(a): return "<64" if a<64 else "64-256" if a<256 else "256-1024" if a<1024 else None
n_img=0; tp={"<64":[],"64-256":[],"256-1024":[]}
for img_id,anns in gt_by_img.items():
    if n_img>=1200: break
    dids=dets_by_img.get(img_id,[])
    if not dids or not anns: n_img+=1; continue
    g_rles=[maskUtils.merge(maskUtils.frPyObjects(g["segmentation"],1024,1024)) for g in anns]
    d_rles=[rle_of(dets[i]) for i in dids]
    iou=maskUtils.iou(d_rles,g_rles,[0]*len(g_rles))
    for lj,g in enumerate(anns):
        b=bucket(g["area"])
        if b is None: continue
        j=int(iou[:,lj].argmax())
        if iou[j,lj]>=0.5: tp[b].append(dets[dids[j]]["score"])
    n_img+=1
for b in ("<64","64-256","256-1024"):
    print(b,"n_tp=",len(tp[b]),"tpmed(first-1200 imgs, best-IoU det)=",round(float(np.median(tp[b])),4) if tp[b] else None)
