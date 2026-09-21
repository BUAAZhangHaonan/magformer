#!/bin/bash
# Compact digest of the f1 full run: progress + latest eval panels + alerts.
PY=/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python
OUT=/home/hdd3/zhanghaonan/magformer/output/aps_20260913
RUN=$OUT/p5_runs/f2_full_design_winners_256k
echo "===== F2 DIGEST $(date '+%m-%d %H:%M') ====="
tail -c 400 "$RUN/run.log" | tr '\r' '\n' | grep -oE "\| [0-9]+/256000 \[[0-9:]+<[0-9:]+, +[0-9.]+s/it" | tail -1
$PY - << 'EOF'
import json
rows = {}
for line in open("/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/f2_full_design_winners_256k/metrics_log.jsonl"):
    d = json.loads(line)
    if d.get("phase") == "val":
        rows[int(d["iter"])] = (d.get("val/segm_AP"), d.get("val/segm_APs"),
                                 d.get("val/segm_APm"), d.get("val/segm_APl"))
for s in sorted(rows)[-3:]:
    print(f"{s:>7}: AP={rows[s][0]:.4f} AP_s={rows[s][1]:.4f} "
          f"AP_m={rows[s][2]:.4f} AP_l={rows[s][3]:.4f}")
import os, glob
snaps = sorted(glob.glob("/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/f2_full_design_winners_256k/eval_snapshots/*/subbucket.json"), key=os.path.getmtime)
if snaps:
    b = json.load(open(snaps[-1]))["buckets"]
    def fmt(k):
        v = b.get(k, {}); med = v.get("tp_score_median")
        return f"{v.get('recall@0.5', 0):.2f}@{med:.2f}" if med is not None else f"{v.get('recall@0.5', 0):.2f}@-"
    print(f"subbucket({os.path.basename(os.path.dirname(snaps[-1]))}): <64:{fmt('<64')} 64-256:{fmt('64-256')} 256-1k:{fmt('256-1024')}")
EOF
echo "checkpoints: $(ls "$RUN"/*.pt 2>/dev/null | wc -l)"
free -g | awk 'NR==2{print "sysmem available:", $7, "GB"}'
