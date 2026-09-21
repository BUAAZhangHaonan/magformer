#!/bin/bash
# F2 pre-experiment probe (Reviewer #3 deliverable, 2026-09-21).
# Purpose: validate from-scratch stability of the FULL winners stack
# (BAS-CL+ / MAL-CP+ / mCDN+ / SCB+ / copy-paste) on the FIXED code and
# measure step-time + memory in both eager and compiled modes, BEFORE the
# 256K launch tonight. Sequential, 2-GPU DDP on GPUs 0+3, wall budget <=1h.
#
#   Arm A  configs/next_stage/f2_probe_winners_700.yaml   eager, 700 steps (~28min)
#   Arm B  configs/next_stage/f2_probe_compile_150.yaml   compiled, 150 steps (~15min)
#
# DO NOT launch while the f2 smoke chain still owns GPUs 0+3 — the
# orchestrator runs this after f2_smoke.sh reports SMOKE CHAIN DONE and
# before B1 frees the slot.
#
# NOTE: no PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here — the
# platform rejects it (see the "expandable_segments not supported" warnings
# in p5_runs/f2_smoke/*.log); exporting it is a silent no-op.
set -u
cd /home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/source
PY=/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python3.11
OUT=/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/f2_probe
mkdir -p "$OUT"

echo "[f2_probe $(date +%H:%M:%S)] arm A: winners-ON EAGER, 700 steps (GPUs 0,3; port 29519)"
$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29519 \
    tools/train.py --config configs/next_stage/f2_probe_winners_700.yaml --gpus 0,3 \
    > "$OUT/winners.log" 2>&1
rc_a=$?
echo "[f2_probe $(date +%H:%M:%S)] arm A exit=$rc_a (log tail below)"
tail -c 800 "$OUT/winners.log" | tr '\r' '\n' | tail -6

echo "[f2_probe $(date +%H:%M:%S)] arm B: winners-ON COMPILED [fusion,rgb,depth], 150 steps (port 29520)"
$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29520 \
    tools/train.py --config configs/next_stage/f2_probe_compile_150.yaml --gpus 0,3 \
    > "$OUT/compile.log" 2>&1
rc_b=$?
echo "[f2_probe $(date +%H:%M:%S)] arm B exit=$rc_b (log tail below)"
tail -c 800 "$OUT/compile.log" | tr '\r' '\n' | tail -6

echo "[f2_probe $(date +%H:%M:%S)] digest (preregistered thresholds)"
$PY - "$OUT" "$rc_a" "$rc_b" <<'PYEOF'
import json, math, statistics as st, sys
from pathlib import Path

out = Path(sys.argv[1]); rc_a = int(sys.argv[2]); rc_b = int(sys.argv[3])
lines = []
kills = []
warns = []
def kill(msg): kills.append(msg)
def warn(msg): warns.append(msg)
def say(msg): lines.append(msg)

# ---------------- Preregistered thresholds ----------------
IT_HARD   = 1.66   # +20% vs A0 1.38 s/it (compiled, 4-GPU)
IT_WARN   = 1.50   # A0 median 1.458; probe runs 2-GPU eager, so this is informational
AMP_SKIP_MAX = 7   # 1% of 700 (A0: 11 skips / 8000 = 0.14%)
CONSEC_MAX   = 8   # trainer hard limit 16; kill well before
GRADNORM_MAX = 6e4 # ~5x A0 observed max 11572 (A0 median 581, p90 2249)
PEAK_ALLOC_MAX = 18500.0   # MB, vs 23500 usable: 4.5GB headroom demanded
MAX_RESERVED_MAX = 23500.0 # MB
REQUIRED_KEYS = ["train/loss_dn_ce", "train/loss_dn_mask", "train/loss_dn_dice",
                 "train/loss_dn_neg_ce", "train/loss_probe", "train/loss_dice_band",
                 "train/loss_entropy", "train/loss_mask", "train/loss_ce"]

def load_rows(sub):
    p = out / sub / "metrics_log.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

# ---------------- Arm A: eager stability ----------------
say("=" * 64)
say("ARM A: winners-ON eager, 700 steps")
say("=" * 64)
if rc_a != 0:
    kill(f"arm A exited rc={rc_a}")
rows = load_rows("winners_700") or []
tr = [r for r in rows if r.get("phase") == "train"]
if not tr:
    kill("arm A produced no train telemetry rows")
else:
    its  = [r["iter_time_sec"] for r in tr if r.get("iter_time_sec")]
    gn   = [r["preclip_grad_norm"] for r in tr if r.get("preclip_grad_norm") is not None]
    cons = [r.get("consecutive_amp_skips", 0) for r in tr]
    amp  = tr[-1].get("amp_skipped_steps", 0)
    peak = max(r.get("peak_memory_mb", 0) for r in tr)
    mrev = max(r.get("cuda_max_memory_reserved_mb", 0) for r in tr)
    scale_end = tr[-1].get("amp_scale")
    # finiteness of every logged loss value (trainer also self-raises)
    bad = []
    for r in tr:
        for k, v in r.items():
            if k.startswith("train/") and isinstance(v, float) and not math.isfinite(v):
                bad.append(f"{k}@{r.get('optimizer_step')}")
    if bad: kill(f"non-finite loss values: {bad[:5]}")
    # amp discipline
    if amp > AMP_SKIP_MAX: kill(f"amp_skipped_steps={amp} > {AMP_SKIP_MAX}")
    if max(cons) >= CONSEC_MAX: kill(f"max consecutive_amp_skips={max(cons)} >= {CONSEC_MAX}")
    if scale_end is not None and scale_end < 2**-10:
        warn(f"amp_scale ended at {scale_end} (<2^-10); A0 settled at 0.0625")
    # grad norm spikes
    if gn and max(gn) > GRADNORM_MAX:
        kill(f"preclip_grad_norm max {max(gn):.0f} > {GRADNORM_MAX:.0f}")
    # winner loss keys present
    missing = [k for k in REQUIRED_KEYS if k not in tr[-1]]
    if missing: kill(f"missing winner loss keys: {missing}")
    # step time (skip first 10 rows = startup/warmup transients)
    steady = its[10:] if len(its) > 12 else its
    med = st.median(steady) if steady else float("nan")
    tail = its[-20:]
    med_tail = st.median(tail) if tail else med
    say(f"rows={len(tr)}  amp_skips={amp}  max_consecutive={max(cons)}  end_scale={scale_end}")
    say(f"grad_norm: median={st.median(gn):.0f}  max={max(gn):.0f}" if gn else "grad_norm: none")
    say(f"iter_time_sec: steady_median={med:.3f}  last200_median={med_tail:.3f}  max={max(its):.3f}")
    say(f"peak_alloc_mb={peak:.0f}  max_reserved_mb={mrev:.0f}")
    if med_tail > IT_HARD:
        kill(f"last-200 median iter_time {med_tail:.3f} > {IT_HARD} (+20% vs A0 1.38)")
    elif med_tail > IT_WARN:
        warn(f"last-200 median iter_time {med_tail:.3f} > {IT_WARN} (investigate before launch)")
    if peak > PEAK_ALLOC_MAX: kill(f"peak alloc {peak:.0f}MB > {PEAK_ALLOC_MAX}MB")
    if mrev > MAX_RESERVED_MAX: kill(f"max reserved {mrev:.0f}MB > {MAX_RESERVED_MAX}MB")
    # trajectory sanity (report-only): first vs last third medians
    def keymed(k, a, b):
        vals = [r[k] for r in tr[a:b] if k in r and isinstance(r[k], float)]
        return st.median(vals) if vals else float("nan")
    n = len(tr)
    for k in ("train/loss_dn_mask", "train/loss_probe", "train/loss_dice_band", "train/loss"):
        f, l = keymed(k, 0, n // 3), keymed(k, 2 * n // 3, n)
        say(f"trajectory {k}: first3rd={f:.4f} last3rd={l:.4f}")
        if not (math.isfinite(f) and math.isfinite(l)):
            kill(f"trajectory {k} non-finite")

# ---------------- Arm B: compile rescue evidence ----------------
say("=" * 64)
say("ARM B: winners-ON compiled [fusion, rgb_backbone, depth_backbone]")
say("=" * 64)
if rc_b != 0:
    kill(f"arm B exited rc={rc_b}")
log = (out / "compile.log").read_text(errors="replace") if (out / "compile.log").exists() else ""
kept_full = "kept: fusion, rgb_backbone, depth_backbone" in log
dropped = [l for l in log.splitlines() if "torch.compile disabled for" in l]
say(f"compile kept full set: {kept_full}  dropped_lines={len(dropped)}")
for l in dropped: say("  " + l.strip()[:160])
rows_b = load_rows("compile_150") or []
trb = [r for r in rows_b if r.get("phase") == "train"]
if trb:
    itsb = [r["iter_time_sec"] for r in trb if r.get("iter_time_sec")]
    medb = st.median(itsb[3:]) if len(itsb) > 4 else (st.median(itsb) if itsb else float("nan"))
    mrevb = max(r.get("cuda_max_memory_reserved_mb", 0) for r in trb)
    say(f"iter_time_sec: median={medb:.3f}  max_reserved_mb={mrevb:.0f}")
    if not kept_full:
        warn("compile fell back (partial or full eager) on 2 ranks too — 256K will run eager; step-time gate then rests on arm A")
else:
    warn("arm B telemetry missing")

say("=" * 64)
say(f"VERDICT: {'KILL — do NOT launch 256K until items below are resolved' if kills else 'PASS — preregistered gates green'}")
for k in kills: say("  [KILL] " + k)
for w in warns: say("  [WARN] " + w)

digest = out / "probe_digest.txt"
digest.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if kills else 0)
PYEOF
dig=$?

echo "[f2_probe $(date +%H:%M:%S)] PROBE CHAIN DONE (armA=$rc_a armB=$rc_b digest=$dig)"
echo "[f2_probe] digest at $OUT/probe_digest.txt"
exit $dig
