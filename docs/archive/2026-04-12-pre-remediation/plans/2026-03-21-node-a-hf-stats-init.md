# Node A Plan: HF Mirror, Dataset Stats, And Initialization

## Scope

- Add Hugging Face mirror support through environment propagation and runner plumbing
- Compute and cache new stats for `20260318_1K_1566`
- Standardize pretrained-weight resolution across MAGFormer, MGM, YOLO, and baseline runners
- Add smoke-friendly config/runner paths for the new dataset

## Deliverables

- dataset stats files for `20260318_1K_1566`
- HF mirror support in scripts/runners
- clear pretrained initialization selection logic
- verification commands and commit

