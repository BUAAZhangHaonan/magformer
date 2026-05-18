# VC-SUDA MagFormer R100-R115 Stage Summary

Date: 2026-05-18
Branch: `feature/vc-suda-sim2real`

## Bottom Line

R100-R115 should be closed as a supervised target-coverage and oracle diagnosis stage.

The main formal held-out/UDA result is still not `61+`. The best formal non-leakage checkpoint is R114:

- R114 val28 segm AP/AP50/AP75: `0.321831 / 0.630752 / 0.295205`.
- R114 remaining75 segm AP/AP50/AP75: `0.392173 / 0.703252 / 0.397419`.

`remaining75` is useful, but it is small. It can move more from individual image composition than the fixed val28 split, so val28 should stay the stable anchor.

The best `61+` number is R115, but it is oracle only:

- R115 train200 oracle segm AP/AP50/AP75: `0.613073 / 0.897043 / 0.720094`.
- R115 val28 segm AP/AP50/AP75: `0.337430 / 0.649099 / 0.323228`.

R115 trains on `target_unlabeled200` ground truth. It must not be reported as formal UDA, held-out target, or paper metric. It is only an upper-bound diagnosis.

## Protocol Guardrail

The old `61+` source sanity and the current target-domain results are different metrics.

The fixed Teacher original first50 protocol gives:

- Teacher original first50 segm AP/AP50/AP75: `0.625293 / 0.867953 / 0.724339`.

That number belongs to the original 1.5K first50 source-sanity eval, not to pseudo-real `target_unlabeled200`, formal held-out target, or UDA.

## Key Technical Conclusions

1. Warm-start is better than true resume for this family.
   R105 true-resumed the full-target oracle path and reached only train200 segm AP `0.366649`. R106 reset optimizer/scheduler/scaler and used model-only warm-start, raising train200 to `0.446421` and val28 to `0.318796`.

2. `importance_sample_ratio=0.0` is a small positive change, not the main fix.
   R107 improves R106 train200 only from `0.446421` to `0.449679`, and val28 from `0.318796` to `0.321195`.

3. Dice weight `10.0` is the best tested dice setting.
   R108 dice `7.5` breaks the R107 plateau. R109 dice `10.0` improves again to train200 `0.476059` and val28 `0.327925`. R110 dice `12.5` goes too far: it improves early train metrics but fails the staged val guard and prediction inflation guards.

4. Target coverage is the main formal driver.
   The formal target path improves as target coverage expands from 25 to 50 to 100 to 150 images. The best formal checkpoint is R114 target150, not the longer R113 target100 true resume.

5. Full target200 oracle proves capacity and recipe are not the hard ceiling.
   With enough target GT, R115 reaches train200 oracle `0.613073`. The current blocker is not simply MagFormer capacity. The blocker is formal semi-supervised or pseudo-label quality under no hidden target GT.

## Metric Table

All values are segm AP/AP50/AP75 under the external 1024 backmap protocol unless noted.

| Run | Target train size | Train AP | Val28 | Held-out / remain | Full200 ref / oracle | Notes |
| --- | ---: | --- | --- | --- | --- | --- |
| R100 | 25 | train25 `0.481398 / 0.824222 / 0.523051` | `0.266639 / 0.588409 / 0.216622` | target_unlabeled200 `0.312443 / 0.635325 / 0.276846` | same as held-out target200 | Clean train-fit recovery from R99. |
| R103 | 25 | train25 `0.527663 / 0.846610 / 0.610345` | `0.272399 / 0.589279 / 0.227188` | target_unlabeled200 `0.314747 / 0.636336 / 0.284706` | same as held-out target200 | True resume to iter0799 gives only a small gain. |
| R104 | 50 | train50 `0.466157 / 0.811844 / 0.495828` | `0.283886 / 0.609918 / 0.231551` | non-overlap175 `0.329132 / 0.656310 / 0.299066` | full200 reference `0.340785 / 0.670643 / 0.316673` | Balanced target50 improves formal validation. Full200 includes promoted train images. |
| R111 | 50 | train50 `0.594213 / 0.896658 / 0.705432` | `0.291622 / 0.603718 / 0.252863` | non-overlap175 `0.339053 / 0.654743 / 0.316917` | full200 reference `0.370225 / 0.683348 / 0.364947` | R109 recipe transfers to formal target50. |
| R112 | 100 | train100 `0.528334 / 0.854196 / 0.600283` | `0.308258 / 0.628971 / 0.267992` | remaining125 `0.365856 / 0.684896 / 0.355083` | full200 reference `0.424739 / 0.747483 / 0.445301` | Coverage expansion gives the clear formal gain. |
| R113 | 100 | train100 `0.561041 / 0.875704 / 0.653386` | `0.305354 / 0.614782 / 0.271655` | remaining125 `0.367070 / 0.679450 / 0.361141` | full200 reference `0.437321 / 0.751328 / 0.468251` | Longer true resume improves train fit, but val28 drops slightly. |
| R114 | 150 | train150 `0.599756 / 0.888771 / 0.708763` | `0.321831 / 0.630752 / 0.295205` | remaining75 `0.392173 / 0.703252 / 0.397419` | full200 reference `0.520191 / 0.820041 / 0.589885` | Best formal non-leakage result. Remaining75 is small. |
| R115 | 200 GT oracle | train200 oracle `0.613073 / 0.897043 / 0.720094` | `0.337430 / 0.649099 / 0.323228` | not held-out target; train uses all target200 GT | full target200 oracle train row `0.613073 / 0.897043 / 0.720094` | Oracle upper bound only. Not formal UDA or held-out. |

## What Not To Report

- Do not report Teacher original first50 `0.625293` as target-domain progress.
- Do not report R115 train200 `0.613073` as formal UDA or held-out target performance.
- Do not compare R114 remaining75 as if it has the same stability as val28 or earlier larger remaining splits.

## Next Step

Stage should be closed.

If work continues, it should move to formal semi-supervised learning and pseudo-label quality. More oracle runs or blind longer training are unlikely to answer the real formal-UDA gap.

## Source Docs Checked

- `docs/results/vc_suda_r44_teacher_first50_protocol_fix_20260516.md`
- `docs/results/baseline_r100_magformer_target_labeled25_cleanfit_20260518.md`
- `docs/results/baseline_r103_magformer_cleanfit_resume_20260518.md`
- `docs/results/baseline_r104_magformer_target_labeled50_balanced_20260518.md`
- `docs/results/baseline_r106_magformer_fulltarget200_oracle_warmstart_20260518.md`
- `docs/results/baseline_r107_magformer_fulltarget200_oracle_randompoints_20260518.md`
- `docs/results/baseline_r108_magformer_fulltarget200_oracle_dice75_20260518.md`
- `docs/results/baseline_r109_magformer_fulltarget200_oracle_dice100_20260518.md`
- `docs/results/baseline_r110_magformer_fulltarget200_oracle_dice125_20260518.md`
- `docs/results/baseline_r111_magformer_r109recipe_target_labeled50_balanced_20260518.md`
- `docs/results/baseline_r112_magformer_r111recipe_target_labeled100_balanced_20260518.md`
- `docs/results/baseline_r113_magformer_r112_target100_resume2000_20260518.md`
- `docs/results/baseline_r114_magformer_r113warm_target150_balanced_20260518.md`
- `docs/results/baseline_r115_magformer_r114warm_fulltarget200_oracle_20260518.md`

Note: this repository has no `docs/metrics` directory at the checked HEAD. The metrics above were verified against the existing tracked docs under `docs/results`.
