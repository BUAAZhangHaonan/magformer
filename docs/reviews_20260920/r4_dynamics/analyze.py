#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R4 独立审阅: 训练动力学与 128K 停训点的机理校验 (CPU only, 只读训练产物)

用法: /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python analyze.py
输出: stdout 的全部表格 (report.md 的数据来源)
"""
import json
import math
import os

import numpy as np

OUT = "/home/hdd3/zhanghaonan/magformer/output/aps_20260913/reviews_20260920/r4_dynamics"

F1 = "/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/f1_full_design_256k/metrics_log.jsonl"
C0C = "/home/hdd3/zhanghaonan/magformer/archive_20260906/staging_4028/home/g203-4028/magformer/output/experiments/next_stage/c0_corrected_300k_seed42/metrics_log.jsonl"
S1C = "/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p4_runs/s1c_design_256k/metrics_log.jsonl"
S0C = "/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p4_runs/s0c_base_256k/metrics_log.jsonl"
ARCH = "/home/hdd3/zhanghaonan/magformer/archive_20260906/staging_4028/home/g203-4028/magformer/output/experiments/next_stage"
OLD_RUNS = ["c0_seed42", "d0_seed42", "p1_seed42", "p2_seed42"]  # 128K cosine 完整跑完的老家族

# ---------------------------------------------------------------- utilities

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def step_of(d):
    for k in ("optimizer_step", "iter"):
        v = d.get(k)
        if isinstance(v, int):
            return v
    return 0


def split_train_val(rows):
    tr = [d for d in rows if d.get("phase") == "train"]
    va = [d for d in rows if d.get("phase") == "val"]
    return tr, va


def cosine_mult(step, max_iter, warmup=2500, warmup_factor=0.001):
    if step < warmup:
        return warmup_factor + (1.0 - warmup_factor) * step / warmup
    progress = (step - warmup) / max(1, max_iter - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def bucket_stats(train_rows, keys, bucket=8000):
    """按 bucket 步长分桶, 返回 {bucket_start: {key: (mean, std, n)}}"""
    out = {}
    for d in train_rows:
        s = step_of(d)
        b = (s // bucket) * bucket
        out.setdefault(b, {}).setdefault("_steps", []).append(s)
        for k in keys:
            v = d.get(k)
            if isinstance(v, (int, float)):
                out[b].setdefault(k, []).append(float(v))
    stats = {}
    for b, agg in sorted(out.items()):
        st = {"n": len(agg.get("_steps", []))}
        for k, vals in agg.items():
            if k == "_steps":
                continue
            a = np.array(vals)
            st[k] = (a.mean(), a.std(), len(a))
        stats[b] = st
    return stats


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------- 1. F1 train 动力学

print("=" * 100)
print("## 1. F1 train 行全量解析 (8K 桶)")
print("=" * 100)

f1_rows = load_jsonl(F1)
f1_tr, f1_va = split_train_val(f1_rows)
print(f"F1: train rows={len(f1_tr)}, val rows={len(f1_va)}, last train step={step_of(f1_tr[-1]) if f1_tr else 0}")

KEYS = ["train/loss", "train/loss_ce", "train/loss_mask", "train/loss_dice",
        "train/loss_bbox", "train/loss_giou", "train/loss_entropy",
        "preclip_grad_norm", "train/lr", "amp_scale"]
stats = bucket_stats(f1_tr, KEYS, 8000)

hdr = ["bucket", "loss", "ce", "mask", "dice", "bbox", "giou", "entropy", "gradnorm(med/p90/clip%)", "lr(max grp)", "amp_scale"]
print(" | ".join(hdr))
for b, st in stats.items():
    def m(k):
        return st.get(k, (float("nan"),) * 3)[0]
    def s(k):
        return st.get(k, (float("nan"),) * 3)[1]
    gn = np.array(st.get("preclip_grad_norm", [np.nan]))
    clip_pct = float(np.mean(gn > 1.0) * 100)
    print(f"{b//1000}K | {m('train/loss'):.2f}±{s('train/loss'):.2f} | "
          f"{m('train/loss_ce'):.4f}±{s('train/loss_ce'):.3f} | "
          f"{m('train/loss_mask'):.4f} | {m('train/loss_dice'):.4f} | "
          f"{m('train/loss_bbox'):.4f} | {m('train/loss_giou'):.4f} | "
          f"{m('train/loss_entropy'):.4f} | "
          f"{np.median(gn):.0f}/{np.percentile(gn,90):.0f}/{clip_pct:.0f}% | "
          f"{m('train/lr'):.2e} | {m('amp_scale'):.3f}")

# 深监督各层 ce/mask/dice (最后一桶 vs 96K 桶: 哪层还在降)
print("\n### 深监督分层 (loss_ce_i / loss_mask_i / loss_dice_i), 关键桶对比")
layer_keys = []
for i in range(7):
    layer_keys += [f"train/loss_ce_{i}", f"train/loss_mask_{i}", f"train/loss_dice_{i}"]
lst = bucket_stats(f1_tr, layer_keys, 8000)
for b in sorted(lst):
    if b < 64000:
        continue
    parts = []
    for i in range(7):
        ce = lst[b].get(f"train/loss_ce_{i}", (np.nan,))[0]
        parts.append(f"L{i}:{ce:.2f}")
    print(f"{b//1000}K ce: " + " ".join(parts))
for b in sorted(lst):
    if b < 64000:
        continue
    parts = []
    for i in range(7):
        mk = lst[b].get(f"train/loss_mask_{i}", (np.nan,))[0]
        parts.append(f"L{i}:{mk:.3f}")
    print(f"{b//1000}K mask: " + " ".join(parts))

# 每分量最近 3 桶 (88K-112K) 相对 72-88K 的下降率
print("\n### 各 loss 分量仍在降的速率 (桶均值, 88K→112K 变化 / 72K→88K 变化)")
for k in ["train/loss", "train/loss_ce", "train/loss_mask", "train/loss_dice", "train/loss_bbox", "train/loss_giou"]:
    v72 = stats.get(72000, {}).get(k, (np.nan,))[0]
    v88 = stats.get(88000, {}).get(k, (np.nan,))[0]
    v104 = stats.get(104000, {}).get(k, (np.nan,))[0]
    print(f"{k:22s}: 72K={v72:.4f} 88K={v88:.4f} 104K={v104:.4f} | Δ(88-104)={v104-v88:+.4f}")

# AMP / 溢出
amp_last = f1_tr[-1].get("amp_skipped_steps", -1)
print(f"\nAMP skip 累计={amp_last} / {step_of(f1_tr[-1])} 步 ({amp_last/max(1,step_of(f1_tr[-1]))*100:.3f}%), "
      f"nonfinite_grad_count={f1_tr[-1].get('nonfinite_grad_count')}, amp_scale={f1_tr[-1].get('amp_scale')}")

# ---------------------------------------------------------------- 2. lr 位置核算 + 两种对齐

print()
print("=" * 100)
print("## 2. lr 位置核算与两种对齐")
print("=" * 100)

print("\n### cosine 乘数与各组 lr (F1: base 1e-5, backbone 0.5x, hires 20x, warmup 2500, max 256000)")
print("step | sched% | mult | main_lr | hires_lr | logged(max)")
for s in [8000, 24000, 48000, 72000, 96000, 112000, 128000, 160000, 192000, 256000]:
    m = cosine_mult(s, 256000)
    logged = None
    near = [d for d in f1_tr if abs(step_of(d) - s) <= 25]
    if near:
        logged = near[len(near) // 2].get("train/lr")
    print(f"{s//1000:>4}K | {100*(s-2500)/253500:5.1f}% | {m:.4f} | {1e-5*m:.2e} | {2e-4*m:.2e} | {logged if logged is not None else float('nan'):.2e}")

# F1 每 8K 增量
f1_aps = {step_of(d): d.get("val/segm_APs") for d in f1_va}
f1_steps = sorted(f1_aps)
print("\n### F1 每 8K AP_s 增量 (绝对步数)")
prev = None
f1_incs = []
for s in f1_steps:
    if prev is not None:
        inc = (f1_aps[s] - f1_aps[prev]) * 100
        f1_incs.append((prev, s, inc))
        print(f"{prev//1000}K→{s//1000}K: {f1_aps[prev]:.4f}→{f1_aps[s]:.4f}  Δ={inc:+.2f}pt "
              f"(sched {100*(prev-2500)/253500:.0f}%→{100*(s-2500)/253500:.0f}%)")
    prev = s

# c0_corrected 增量 (37.5K 窗口)
c0_rows = load_jsonl(C0C)
c0_tr, c0_va = split_train_val(c0_rows)
c0_aps = {step_of(d): d.get("val/segm_APs") for d in c0_va}
print(f"\n### c0_corrected_300k: train rows={len(c0_tr)}, last train={step_of(c0_tr[-1])}, val points={sorted(c0_aps)}")
print("  → 关键: eval_period=37500, 最后一次 val=112.5K, 下一次本应在 150K — 112.5K→141K 段【无任何 AP_s 测量】")
prev = None
for s in sorted(c0_aps):
    if prev is not None:
        inc = (c0_aps[s] - c0_aps[prev]) * 100
        print(f"{prev//1000}K→{s//1000}K: Δ={inc:+.2f}pt/37.5K = {inc/4.6875:+.2f}pt/8K (sched {100*(prev-2500)/297500:.0f}%→{100*(s-2500)/297500:.0f}%)")
    prev = s

# c0_corrected train loss 112.5→141K 是否停滞
c0_stats = bucket_stats(c0_tr, ["train/loss", "train/loss_ce", "train/loss_mask", "train/loss_dice", "train/lr"], 8000)
print("\n### c0_corrected train loss 8K 桶 (112.5K→141K 停训段)")
for b, st in sorted(c0_stats.items()):
    if b >= 88000:
        print(f"{b//1000}K | loss={st.get('train/loss',(float('nan'),))[0]:.2f}±{st.get('train/loss',(0,float('nan')))[1]:.2f} | "
              f"ce={st.get('train/loss_ce',(float('nan'),))[0]:.4f} | mask={st.get('train/loss_mask',(float('nan'),))[0]:.4f} | "
              f"dice={st.get('train/loss_dice',(float('nan'),))[0]:.4f} | lr={st.get('train/lr',(float('nan'),))[0]:.2e}")

# 小模型臂增量
print("\n### s1c/s0c 每 8K AP_s 增量 (256K cosine 同日程)")
for name, path in [("s1c", S1C), ("s0c", S0C)]:
    rows = load_jsonl(path)
    _, va = split_train_val(rows)
    aps = {step_of(d): d.get("val/segm_APs") for d in va}
    prev = None
    line = []
    for s in sorted(aps):
        if prev is not None:
            line.append(f"{prev//1000}→{s//1000}:{(aps[s]-aps[prev])*100:+.2f}")
        prev = s
    last_tr = max(step_of(d) for d in rows)
    print(f"{name} (last_train={last_tr}): " + " ".join(line))

# 老 128K 完整家族: 后半程收益
# 注意单位: 老家族 grad_accum=4, metrics 记的 optimizer_step 总量 = 128K micro / 4 = 32K;
# F1/c0c/s 臂 grad_accum=1 (DDP 4卡), optimizer_step 总量 = max_iter 原值。
# 两边 optimizer_step↔数据量等价 (每 optimizer step 4 图), 日程% 用各自 optimizer_step 总量归一。
print("\n### 老 128K cosine 完整跑 (老家族, eval 16K micro = 4K optimizer): 后半程 tail gain")
old_data = {}
for name in OLD_RUNS:
    p = os.path.join(ARCH, name, "metrics_log.jsonl")
    if not os.path.isfile(p):
        continue
    rows = load_jsonl(p)
    _, va = split_train_val(rows)
    aps = {step_of(d): d.get("val/segm_APs") for d in va}
    if len(aps) < 5:
        continue
    old_data[name] = aps
    ss = sorted(aps)
    total_sched = 32000  # 128K micro / grad_accum 4
    print(f"{name}: " + " ".join(f"{s}ops({100*s/total_sched:.0f}%):{aps[s]:.4f}" for s in ss))
    half = max([s for s in ss if s <= total_sched * 0.5], default=None)
    q3 = max([s for s in ss if s <= total_sched * 0.75], default=None)
    end_s, end_v = ss[-1], aps[ss[-1]]
    first_v = aps[ss[0]]
    if half and q3:
        print(f"   后半({100*half/total_sched:.0f}%→{100*end_s/total_sched:.0f}%)={(end_v-aps[half])*100:+.2f}pt "
              f"({(end_v-aps[half])/(end_v-first_v)*100:.0f}% of total) | "
              f"末1/4({100*q3/total_sched:.0f}%→end)={(end_v-aps[q3])*100:+.2f}pt")

# ---------------------------------------------------------------- 3. EMA 分析

print()
print("=" * 100)
print("## 3. EMA 0.9999 分析")
print("=" * 100)
d = 0.9999
hl = math.log(2) / (1 - d)
mean_lag = d / (1 - d)
print(f"半衰期 = ln2/(1-d) = {hl:.0f} 步; 指数核均值滞后 = d/(1-d) = {mean_lag:.0f} 步 (~{mean_lag/8000:.2f} 个 eval 窗口)")
print("trainer.py 已核实: evaluate() 用 ema.apply_shadow → val 全部是 EMA 权重; best.pt 存 EMA(+raw backup)。")

# 增量噪声带: 最近 40K 的增量对平滑拟合的残差
incs = np.array([x[2] for x in f1_incs])
steps_c = np.array([(x[0] + x[1]) / 2 for x in f1_incs])
sel = steps_c >= 56000
# 对 ln(step) 线性回归增量
A = np.vstack([np.ones(sel.sum()), np.log(steps_c[sel])]).T
coef, *_ = np.linalg.lstsq(A, incs[sel], rcond=None)
resid = incs[sel] - A @ coef
print(f"\n最近 7 窗 (56K-112K) 增量 = {np.round(incs[sel],2)}")
print(f"对 ln(step) 回归后残差 std = {resid.std():.2f}pt → 单窗增量测量噪声 ~±{resid.std():.2f}pt")
print(f"最近 3 窗平均 = {incs[-3:].mean():.2f}pt/8K (vs 单窗 104→112K 的 {incs[-1]:.2f}pt — 后者受噪声支配)")

# EMA 滞后对增量的影响 (线性趋势下增量一阶不变, 只有时移)
rate_16k = (f1_aps[112000] - f1_aps[96000]) * 100  # pt / 16K 步
lag_pt = rate_16k / 16000 * mean_lag
print(f"\n当前趋势速率 ~{rate_16k:.2f}pt/16K → EMA 水平滞后(线性近似) ≈ {lag_pt:.2f}pt; "
      f"增量一阶抵消, 只有时移 ~{mean_lag/8000:.1f} 窗")
print("方向性结论: 减速曲线下 EMA 增量[104,112]K 实际反映的是 ~[94,102]K 的真实改善速率 — 当前瞬时速率只会更低, EMA 不会制造虚假的平台。")

# train loss 24K 后斜率显著性 (桶均值间有共同慢漂移, 桶内 sem 低估噪声 → 用无权重 OLS + 残差 stderr)
print("\n### train loss 分量 24K→112K 线性斜率 (桶均值无权重 OLS, 残差 stderr)")
selb = [b for b in sorted(stats) if b >= 24000]
xb = np.array(selb, dtype=float)
for k in ["train/loss", "train/loss_ce", "train/loss_mask", "train/loss_dice", "train/loss_giou"]:
    yb = np.array([stats[b][k][0] for b in selb])
    A5 = np.vstack([np.ones(len(xb)), xb]).T
    c5, *_ = np.linalg.lstsq(A5, yb, rcond=None)
    resid = yb - A5 @ c5
    dof = len(xb) - 2
    s2 = (resid ** 2).sum() / dof
    cov = s2 * np.linalg.inv(A5.T @ A5)
    se_slope = math.sqrt(cov[1, 1])
    per8k = c5[1] * 8000
    se8 = se_slope * 8000
    print(f"{k:18s}: slope = {per8k:+.4f} ± {se8:.4f} /8K (t={per8k/se8:+.1f}) → "
          f"{'显著下降' if per8k < -2*se8 else '统计上不降 (平台)'}, 残差std={resid.std():.4f}")

# ---------------------------------------------------------------- 4. 外推

print()
print("=" * 100)
print("## 4. 128K→256K AP_s 外推")
print("=" * 100)

# (a) 幂律拟合增量: inc = A * s^-p  (从 24K 起, 避开 warmup 段)
sel2 = steps_c >= 20000
x = np.log(steps_c[sel2])
y = np.log(np.maximum(incs[sel2], 1e-3))
A2 = np.vstack([np.ones(sel2.sum()), x]).T
c2, *_ = np.linalg.lstsq(A2, y, rcond=None)
AA, pp = math.exp(c2[0]), -c2[1]
fut_centers = np.array([(128000 + i * 8000 + 4000) for i in range(16)])
inc_powlaw = AA * fut_centers ** (-pp)
gain_powlaw = inc_powlaw.sum()
print(f"(a) 幂律增量拟合 (≥20K): inc = {AA:.2e}·s^-{pp:.3f}  → 128K→256K 总增益 = {gain_powlaw:.2f}pt")

# (b) 对数曲线拟合 AP_s(s) = a + b ln s (56K-112K)
sel3 = np.array([s for s in f1_steps if s >= 56000], dtype=float)
y3 = np.array([f1_aps[int(s)] for s in sel3])
A3 = np.vstack([np.ones(len(sel3)), np.log(sel3)]).T
c3, *_ = np.linalg.lstsq(A3, y3, rcond=None)
a3, b3 = c3
gain_log = (a3 + b3 * math.log(256000) - f1_aps[112000]) * 100
gain_log128 = (a3 + b3 * math.log(256000) - (a3 + b3 * math.log(128000))) * 100
print(f"(b) 对数拟合 (≥56K): AP_s = {a3:.4f} + {b3:.5f}·ln(s) → 112K→256K 增益 = {gain_log:.2f}pt; 128K→256K = {gain_log128:.2f}pt")

# (c) 老 128K 家族 schedule 相对形状锚
print("(c) 老 128K 家族锚: 后半程增益 +0.71~+0.83pt, 末 1/4 +0.24~+0.26pt (绝对 pt)")
print(f"    F1 在 sched 40% 处的实测速率 (+0.21pt/8K, 3窗均值) 与老家族同位置 (~+0.21pt/8K) 一致 → 形状迁移给乐观上界 ~+0.8pt")
print(f"    悲观下界: c0_corrected 式塌缩 (112.5K 后无测量, 但 37.5% 处已降至 +0.14pt/8K) → +0~0.2pt")

# (d) 场景汇总 (以 128K 停训点为基准)
recent_rate = incs[-3:].mean()
ap128_est = f1_aps[112000] + recent_rate / 100 * 2  # 112→128K 两窗
ap128_lo = f1_aps[112000] + incs[-1] / 100 * 2      # 若按最差单窗
ap128_hi = f1_aps[112000] + max(incs[-4:]) / 100 * 2
print(f"\n128K 落点估计: 0.2638 + 2×{recent_rate:.2f}pt ≈ {ap128_est:.4f} (区间 {ap128_lo:.4f}~{ap128_hi:.4f}, EMA val 口径)")
scenarios = {
    "悲观 (增量→噪声底 ±0.05pt/8K)": 0.1,
    "中心 (幂律/对数/老家族锚 三模型取中位)": float(np.median([gain_powlaw, gain_log128, 0.83])),
    "乐观 (对数曲线外推)": gain_log128,
}
for k, v in scenarios.items():
    print(f"  {k}: 128K→256K 增益 {v:+.2f}pt → 256K 终点 ~{ap128_est + v/100:.4f}")

# 4b. 附: 全 mAP 的同款外推 (参考)
f1_map = {step_of(x): x.get("val/segm_AP") for x in f1_va}
sel4 = np.array([s for s in f1_steps if s >= 56000], dtype=float)
y4 = np.array([f1_map[int(s)] for s in sel4])
A4 = np.vstack([np.ones(len(sel4)), np.log(sel4)]).T
c4, *_ = np.linalg.lstsq(A4, y4, rcond=None)
print(f"参考: mAP 对数拟合 → 128K→256K 全 mAP 增益 {(c4[0]+c4[1]*math.log(256000)-(c4[0]+c4[1]*math.log(128000)))*100:+.2f}pt")

# 5. 续训对照臂的成本核算
last = f1_tr[-1]
el = last.get("elapsed_sec", 0)
st = step_of(last)
spd = st / el
print(f"\n### 续训对照成本: 当前 {st} 步 / {el/3600:.1f}h = {spd*3600/1000:.1f}K 步/小时 (含 eval 开销)")
print(f"    128K→160K (32K 步 + 4 次 eval) ≈ {32000/spd/3600:.1f}h ; 128K→144K ≈ {16000/spd/3600:.1f}h ; 完整 128K→256K ≈ {128000/spd/3600:.1f}h")

print("\n[done]")
