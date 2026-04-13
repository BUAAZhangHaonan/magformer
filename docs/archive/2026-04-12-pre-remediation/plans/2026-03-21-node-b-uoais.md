# Node B Plan: UOAIS Compatibility Repair

## Scope

- Repair UOAIS for the current Torch/Detectron2 environment
- Avoid dead old CUDA-op assumptions where possible
- Make ECC dataset training/eval path work on `20260318_1K_1566`
- Prove it with a short smoke train/eval run

## Deliverables

- repaired UOAIS runner/import path
- smoke-tested `uoais` training command
- node verification, commit, and push

