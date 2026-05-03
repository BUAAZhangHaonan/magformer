# Node D Plan: DDP And Representative Smoke Validation

## Scope

- Enable robust 2-GPU DDP flows on the current dual-A100 host
- Validate representative training paths across model families
- Use short smoke runs to confirm startup, forward/backward, checkpointing, and metrics output

## Deliverables

- DDP-capable launcher/runners
- representative smoke jobs for MAGFormer, MGM, YOLO, UOAIS, and U-Net
- verification evidence and node commit/push

