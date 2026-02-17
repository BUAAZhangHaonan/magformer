# Final 20k Dual-Track Validation Report (Running)

Date: February 18, 2026  
Scope: M10 (`exp(final): complete 20k dual-track validation and final report`)

## 1) Current Status

- Status: `RUNNING`
- Launch time: `2026-02-18 05:39:45 CST`
- Launcher script: `scripts/experiments/run_dual_track_20k.sh`
- PID: `2656384` (saved in `output/experiments/final_20k/pid.txt`)
- Log file: `output/experiments/final_20k/run.log`
- Active stage: `Track A 20k / MagFormer training`
- Latest observed progress snapshot:
  - `output/experiments/track_a_20k/magformer/metrics_log.jsonl`
  - `train iter=20`

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
