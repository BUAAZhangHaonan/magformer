import json, time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
GT = "/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/annotations/instances_val.validated.json"
DET = "/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/f1_full_design_256k/eval_snapshots/2001_0920/dets.json"
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    coco = COCO(GT)
    dets = json.load(open(DET))
    coco_dt = coco.loadRes(dets)
t0=time.time()
e = COCOeval(coco, coco_dt, "segm")
e.evaluate(); e.accumulate()
import numpy as np
p = e.params
print("default iouThrs:", p.iouThrs)
s = e.stats
print("plain COCOeval stats AP,AP50,AP75,APs,APm,APl,AR1,AR10,AR100,ARs,ARm,ARl:")
print([round(float(x),6) for x in s])
print("eval_s", round(time.time()-t0,1))
# now with round() grid like trainer
e2 = COCOeval(coco, coco_dt, "segm")
e2.params.iouThrs = np.asarray([round(0.50+0.05*i,2) for i in range(10)], dtype=np.float64)
e2.evaluate(); e2.accumulate(); e2.summarize()
print("rounded-grid stats:", [round(float(x),6) for x in e2.stats])
