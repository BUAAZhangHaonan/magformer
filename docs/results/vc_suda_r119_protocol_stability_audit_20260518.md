# R119 No-Train Protocol Stability Audit

Date: 2026-05-18
Branch: `feature/vc-suda-sim2real`
Mode: no training, no new inference. The audit reads existing COCO GT, predictions, and metrics only.

## Scope

R119 checks whether the current formal evaluation protocol is stable enough to justify more formal `61+` target training experiments.

Inputs used:

- R114 formal best val28: `0.321831` segm AP.
- R114 formal best remaining75: `0.392173` segm AP.
- R114 full200 reference: `0.520191` segm AP.
- R115 train200 oracle: `0.613073` segm AP.
- R118 small-step Stage C metrics as failure evidence.

Diagnostic output:

- `output/diagnostics/r119_protocol_stability_20260518/summary.json`
- `output/diagnostics/r119_protocol_stability_20260518/summary.md`
- `output/diagnostics/r119_protocol_stability_20260518/per_image_proxy.csv`

The output directory is generated evidence and is not meant to be committed.

## Method

`tools/diagnose_protocol_stability.py` recomputes exact COCO AP for the existing predictions and GT with `maxDets=200`, then checks the recomputed metrics against the saved `metrics.cocoeval.json` files.

It also computes:

- density buckets by per-image GT count: `low25 <= 25`, `mid50 <= 50`, `high100 > 50`;
- area buckets by GT mask area: `tiny_area_le_256`, bottom 20%, middle 60%, top 20%;
- per-image matched segm AP proxy, then a 1000-sample bootstrap CI over images;
- label-count trend between image annotation count and per-image AP proxy.

The bootstrap CI is a proxy, not official COCO bootstrap AP. Official COCO AP is global over the full ranked prediction list, so exact replacement bootstrap would require many cloned-image COCOeval runs. R119 still uses exact COCO AP for the headline and bucket rows, and uses the proxy CI only to estimate split sensitivity.

## Headline Metrics

| run | images | annotations | segm AP/AP50/AP75 | proxy AP 95% CI | count-vs-proxy r |
| --- | ---: | ---: | ---: | ---: | ---: |
| R114 val28 | 28 | 1892 | `0.321831 / 0.630752 / 0.295205` | `0.363449 [0.312942, 0.412291]` | `-0.873750` |
| R114 remaining75 | 75 | 4364 | `0.392173 / 0.703252 / 0.397419` | `0.450419 [0.412869, 0.486361]` | `-0.819368` |
| R114 full200 ref | 200 | 11750 | `0.520191 / 0.820041 / 0.589885` | `0.574640 [0.550798, 0.597255]` | `-0.714192` |
| R115 train200 oracle | 200 | 11750 | `0.613073 / 0.897043 / 0.720094` | `0.661594 [0.643729, 0.679203]` | `-0.894830` |

The proxy bootstrap shows that val28 is visibly wide because it has only 28 images. remaining75 is better, but still small. The 200-image rows are much tighter. All four rows show a strong negative label-count trend, which means denser images are consistently harder.

## Sample And Density

| split | images | annotations | per-image count min/p25/median/p75/max | density distribution |
| --- | ---: | ---: | --- | --- |
| R114 val28 | 28 | 1892 | `50 / 50 / 50 / 98.25 / 100` | `mid50: 18 images / 900 anns`, `high100: 10 images / 992 anns` |
| R114 remaining75 | 75 | 4364 | `25 / 50 / 50 / 50 / 100` | `low25: 9 images / 225 anns`, `mid50: 49 images / 2448 anns`, `high100: 17 images / 1691 anns` |
| full200 target GT | 200 | 11750 | `25 / 50 / 50 / 50 / 100` | `low25: 22 images / 550 anns`, `mid50: 131 images / 6547 anns`, `high100: 47 images / 4653 anns` |

val28 has no low-density images. More than half of its annotations are in high-density images, even though high-density is only 10 of 28 images. This makes val28 useful as a hard anchor, but not a stable estimate of average target-domain quality by itself.

## Density Buckets

| run | low25 segm AP | mid50 segm AP | high100 segm AP |
| --- | ---: | ---: | ---: |
| R114 val28 | n/a | `0.442028` | `0.205894` |
| R114 remaining75 | `0.721433` | `0.466071` | `0.230508` |
| R114 full200 ref | `0.782532` | `0.604628` | `0.362732` |
| R115 train200 oracle | `0.833687` | `0.699180` | `0.456716` |

High-density images dominate the formal error. In R114 remaining75, high100 is `0.230508`, about half of mid50 and far below low25. The same shape remains in the oracle: train200 reaches `0.613073` overall, but high100 is still only `0.456716`.

## Area Buckets

| run | tiny <=256 AP | bottom20 AP | mid60 AP | top20 AP |
| --- | ---: | ---: | ---: | ---: |
| R114 val28 | `0.014453` | `0.000967` | `0.329726` | `0.554395` |
| R114 remaining75 | `0.019607` | `0.008169` | `0.423903` | `0.602704` |
| R114 full200 ref | `0.110068` | `0.067496` | `0.575601` | `0.721152` |
| R115 train200 oracle | `0.208758` | `0.150236` | `0.681512` | `0.804920` |

Small targets are the clearest failure mode. For R114 remaining75, bottom20 area AP is `0.008169`, while top20 area AP is `0.602704`. Oracle training helps every bucket, but the tiny and bottom20 buckets stay much weaker than mid/top area buckets.

## R118 Stage C Evidence

R118 small-step used the R114 warm teacher and the R117 pseudo bank without new inference. It failed the iter100 gate:

| split | R118 segm AP/AP50/AP75 | gate result |
| --- | ---: | --- |
| val28 | `0.321894 / 0.636520 / 0.293118` | below `0.322` |
| remaining75 | `0.392568 / 0.706452 / 0.395474` | below `0.397` |
| full200 ref | `0.520202 / 0.820947 / 0.587653` | below `0.523` |

This does not show source collapse. It shows that the current Stage C path is not creating meaningful target-side improvement over R114.

## Oracle Boundary

R115 train200 oracle is a capacity and recipe bound, not a formal metric. It trains on all target200 GT, so `0.613073` cannot be reported as UDA, held-out target, or non-leakage performance.

The formal non-leakage boundary remains R114 val28 and R114 remaining75:

- R114 val28: `0.321831`.
- R114 remaining75: `0.392173`.

R114 full200 reference is useful for continuity, but it includes promoted training images and is not a clean held-out number. The gap between R115 oracle train200 `0.613073` and formal held-out rows is therefore a label/protocol boundary, not evidence that more no-change Stage C training is likely to reach formal `61+`.

## Decision

Pause formal `61+` training experiments.

Reasons:

1. The formal held-out anchors are far below `61+`: val28 `0.321831`, remaining75 `0.392173`.
2. The hardest buckets are structural: high-density and tiny-area AP remain weak even under the oracle.
3. R118 Stage C already failed the small-step gate without improving target quality.

The right next step is not more Stage C. It should be one of:

- close this line and report R114 as the formal best plus R115 as oracle only;
- add a larger stable held-out test or more labels for the high-density and tiny-object cases;
- do a structure-level diagnosis focused on dense small masks, matching, and mask resolution.

## Validation

Commands run:

```bash
python tools/diagnose_protocol_stability.py --self-test
python tools/diagnose_protocol_stability.py --out-dir output/diagnostics/r119_protocol_stability_20260518 --bootstrap-samples 1000 --run r114_val28 --run r114_remaining75 --run r114_full200_reference --run r115_train200_oracle
```

The diagnostic run completed and the script checked recomputed overall COCO metrics against the saved metrics files with tolerance `1e-9`.
