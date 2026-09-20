#!/usr/bin/env python3
"""128K pause-and-seal watcher for f1_full_design_256k (user decision 2026-09-20).

Waits for the 128000-step val entry in metrics_log.jsonl (checkpoint last.pt is
saved before the eval runs), then SIGTERMs the torchrun tree, stops the eval
snapshot watcher, and seals weights/metrics/dets/config into p5_runs/f1_seal_128k.
Runs detached on the server so it survives client closure.
"""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time

RUN = "/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/f1_full_design_256k"
SEAL = "/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/f1_seal_128k"
DIGEST = "/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/digest_history.log"
CFG = ("/home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/"
       "source/configs/aps_20260913_full/f1_full_design_256k.yaml")
TARGET = 128000


def ts():
    return time.strftime("%m-%d %H:%M:%S")


def log(msg):
    line = "[{}] SEAL-WATCH: {}".format(ts(), msg)
    print(line, flush=True)
    with open(DIGEST, "a") as f:
        f.write(line + "\n")


def trainer_pids():
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    pids = []
    for ln in out.splitlines():
        if "f1_full_design_256k" not in ln:
            continue
        if "tools/train.py" not in ln and "torchrun" not in ln:
            continue
        if "grep" in ln or "seal" in ln:
            continue
        head = ln.split(None, 1)
        if len(head) == 2:
            try:
                pids.append(int(head[0]))
            except ValueError:
                pass
    return pids


def val_reached():
    try:
        with open(os.path.join(RUN, "metrics_log.jsonl")) as f:
            for line in f:
                if '"val"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("phase") == "val" and int(d.get("iter", 0)) >= TARGET:
                    return True
    except FileNotFoundError:
        pass
    return False


def latest_val():
    final = None
    try:
        with open(os.path.join(RUN, "metrics_log.jsonl")) as f:
            for line in f:
                if '"val"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("phase") == "val":
                    final = d
    except FileNotFoundError:
        pass
    return final


log("armed; waiting for val@{} (trainer pids now: {})".format(TARGET, trainer_pids()))
while True:
    if val_reached():
        break
    if not trainer_pids():
        log("WARNING: trainer gone before val@{}; sealing whatever exists".format(TARGET))
        break
    time.sleep(120)

log("val@{} observed (or trainer gone); 300s grace for best.pt/dets writes".format(TARGET))
time.sleep(300)

pids = trainer_pids()
if pids:
    log("SIGTERM {}".format(pids))
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(90)
    left = trainer_pids()
    if left:
        log("SIGKILL {}".format(left))
        for p in left:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(30)
else:
    log("trainer already stopped")

# stop the eval snapshot watcher (no more evals); keep the hourly digest daemon
subprocess.run(
    ["bash", "-c",
     "ps -eo pid,args | grep 'f1_watch.sh' | grep -v grep | awk '{print $1}' | xargs -r kill -TERM"],
    capture_output=True)

os.makedirs(SEAL, exist_ok=True)
for name in ["last.pt", "best.pt", "coco_instances_results.json",
             "metrics_log.jsonl", "metrics_log.csv"]:
    src = os.path.join(RUN, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(SEAL, name))
        log("sealed {}".format(name))
if os.path.exists(CFG):
    shutil.copy2(CFG, os.path.join(SEAL, "f1_full_design_256k.yaml"))

snap_root = os.path.join(RUN, "eval_snapshots")
if os.path.isdir(snap_root):
    snaps = [os.path.join(snap_root, d) for d in os.listdir(snap_root)]
    snaps = [s for s in snaps if os.path.isdir(s)]
    if snaps:
        latest = max(snaps, key=os.path.getmtime)
        shutil.copytree(latest, os.path.join(SEAL, "final_eval_snapshot"), dirs_exist_ok=True)
        log("sealed snapshot {}".format(latest))

runlog = os.path.join(RUN, "run.log")
if os.path.exists(runlog):
    with open(runlog, "rb") as f, open(os.path.join(SEAL, "run.log.tail"), "wb") as g:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 20 * 1024 * 1024))
        shutil.copyfileobj(f, g)

with open(os.path.join(SEAL, "SHA256SUMS"), "w") as sums:
    for name in ["last.pt", "best.pt"]:
        p = os.path.join(SEAL, name)
        if os.path.exists(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            sums.write("{}  {}\n".format(h.hexdigest(), name))

gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv", "noheader"],
    capture_output=True, text=True).stdout
with open(os.path.join(SEAL, "gpu_after_stop.txt"), "w") as f:
    f.write(gpu)

final = latest_val()
lines = ["# F1 128K seal note",
         "sealed at {} per user decision 2026-09-20: stop at 128K, archive weights, "
         "then test whether new algorithms reactivate AP_s growth on this base.".format(ts()),
         ""]
if final:
    lines.append("final val entry (iter {}):".format(final.get("iter")))
    for k in sorted(final):
        if k.startswith("val/"):
            lines.append("  {} = {}".format(k, final.get(k)))
else:
    lines.append("no val entry found")
lines += ["", "resume path: set runtime.resume={}/last.pt in a copy of the config".format(SEAL),
          "gpu after stop:", gpu]
with open(os.path.join(SEAL, "SEAL_NOTES.md"), "w") as f:
    f.write("\n".join(lines) + "\n")

with open(os.path.join(os.path.dirname(SEAL), "SEAL_128K_DONE.txt"), "w") as f:
    f.write("sealed at {} -> {}\n".format(ts(), SEAL))
log("SEALED -> {}; GPUs 4-7 free for next phase".format(SEAL))
