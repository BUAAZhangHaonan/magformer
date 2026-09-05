import torch
import sys
sys.path.insert(0, ".")

from magformer.config.loader import load_config
from magformer.models.magformer.arch import MagFormerArch

# Build v99 model (MBV3-L depth backbone)
cfg99 = load_config("configs/v99_mbv3l_8dec_128k_from_v88.yaml")
model99 = MagFormerArch.from_config(cfg99)
p99 = sum(p.numel() for p in model99.parameters())
print(f"v99 (MBV3-L) total params: {p99:,}")

if hasattr(model99, "depth_backbone"):
    db99 = sum(p.numel() for p in model99.depth_backbone.parameters())
    print(f"v99 depth_backbone params: {db99:,}")

if hasattr(model99, "backbone"):
    bb99 = sum(p.numel() for p in model99.backbone.parameters())
    print(f"v99 RGB backbone params: {bb99:,}")

print("\n--- v99 all children ---")
for name, mod in model99.named_children():
    n = sum(p.numel() for p in mod.parameters())
    print(f"  {name}: {n:,}")

# Build v103 model (ConvNeXt-T depth backbone)
cfg103 = load_config("configs/v103_convnextT_8dec_128k_from_v101.yaml")
model103 = MagFormerArch.from_config(cfg103)
p103 = sum(p.numel() for p in model103.parameters())
print(f"\nv103 (ConvNeXt-T) total params: {p103:,}")

if hasattr(model103, "depth_backbone"):
    db103 = sum(p.numel() for p in model103.depth_backbone.parameters())
    print(f"v103 depth_backbone params: {db103:,}")

if hasattr(model103, "backbone"):
    bb103 = sum(p.numel() for p in model103.backbone.parameters())
    print(f"v103 RGB backbone params: {bb103:,}")

print("\n--- v103 all children ---")
for name, mod in model103.named_children():
    n = sum(p.numel() for p in mod.parameters())
    print(f"  {name}: {n:,}")

print(f"\nTotal param delta (v103 - v99): {p103 - p99:,}")
