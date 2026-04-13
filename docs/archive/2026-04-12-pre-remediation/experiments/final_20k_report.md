# **⚠️ This document is superseded and retained for historical reference only. Do not use its numbers as current results.**

# Final 20k Dual-Track Validation Report (Paused)

Date: February 18, 2026  
Scope: M10 (`exp(final): complete 20k dual-track validation and final report`)

## 1) Current Status

- Status: `PAUSED_FOR_BASELINE_SUITE`
- Launch time: `2026-02-18 05:39:45 CST`
- Launcher script: `scripts/experiments/run_dual_track_20k.sh`
- PID: `2656384` (saved in `output/experiments/final_20k/pid.txt`)
- Log file: `output/experiments/final_20k/run.log`
- Stop time: `2026-02-18 19:27:46 CST`
- Stop action: sent `SIGINT` to runner + child training processes to free `GPU0` for the baseline suite
- PIDs at stop (for reference):
  - dual-track runner: `2656384`
  - track-b runner: `3030184`
  - mask2former (conda wrapper): `3069129`
  - mask2former (train proc): `3069156`
- Active stage at stop: `Track B 20k / Mask2Former training` (MagFormer stage completed first; then Track B started Mask2Former)
- Latest observed progress snapshot:
  - Track A (completed):
    - Summary: `output/experiments/track_a_20k/track_a_20k_summary.json`
    - MagFormer artifacts: `output/experiments/track_a_20k/magformer/`
    - Mask2Former artifacts: `output/experiments/track_a_20k/mask2former/`
  - Track B (interrupted/paused):
    - MagFormer: `output/experiments/track_b_20k/magformer/metrics_log.jsonl` (latest train iter observed: `720`)
    - Mask2Former: `output/experiments/track_b_20k/mask2former/metrics.json` (latest iter observed: `179`)
    - Note: Track B checkpoint/eval periods are large (`checkpoint_period=5000`, `eval_period=2000`), so an early stop may not have a recent checkpoint.

## 2) Run Commands

- Dry-run check:
  - `bash scripts/experiments/run_dual_track_20k.sh --dry-run`
- Full launch:
  - `nohup bash scripts/experiments/run_dual_track_20k.sh --run > output/experiments/final_20k/run.log 2>&1 &`

## 3) Expected Outputs (after completion)

- Track A 20k summary: `output/experiments/track_a_20k/track_a_20k_summary.json`
- Track B 20k summary: `output/experiments/track_b_20k/track_b_20k_summary.json`
- Final merged summary: `output/experiments/final_20k/final_20k_summary.json`

## 4) 2k Reference Snapshot (Completed)

- Track A 2k report: `docs/experiments/track_a_2k_report.md`
- Track B 2k report: `docs/experiments/track_b_2k_report.md`
- Track B 2k key result (best/last, unified summary):
  - MagFormer AP: `75.05`
  - Mask2Former AP: `86.23`
  - Gap (Mag - M2F): `-11.18`

## 5) To Be Filled After 20k Completion

- [ ] Track A 20k AP/AP50/AP75/APs/APm table
- [ ] Track B 20k AP/AP50/AP75/APs/APm table
- [ ] Best/last gap analysis between MagFormer and Mask2Former
- [ ] Final conclusion on "追平/反超"
- [ ] Error analysis and representative triptych comparisons
