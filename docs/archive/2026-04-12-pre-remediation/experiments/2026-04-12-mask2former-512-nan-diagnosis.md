# Mask2Former 512 NaN Diagnosis

## Conclusion

The existing repo evidence points to a baseline-side evaluation warning, not a project-owned metric bug. The warning
text comes from vendored Detectron2 evaluation code in the 512 Mask2Former experiment log, where `d2.evaluation.coco_evaluation`
and `d2.evaluation.testing` emit `Some metrics cannot be computed and is shown as NaN`.

## Evidence

- The 512 Mask2Former log shows the NaN warning in Detectron2 evaluation, not in MAGFormer project code.
- The run still reaches a final checkpoint save, so this is a metric-computation warning path rather than a total run failure.
- The live metrics row for the run is populated, so the current evidence does not prove a project-owned evaluation regression.

## Action

Treat the 512 Mask2Former NaN warning as archival baseline behavior unless a later project-owned code path is shown to trigger it.
