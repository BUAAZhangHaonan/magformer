import sys
sys.path.insert(0, '/home/g203-4028/magformer')
from magformer.config import load_config
from magformer.models.build import build_model

cfg = load_config('/home/g203-4028/magformer/configs/v84_pruned_32k.yaml')
model = build_model(cfg)
model.cuda()
model.eval()

total = 0
components = {}

for name, module in model.named_children():
    count = sum(p.numel() for p in module.parameters())
    components[name] = count
    total += count

print("=== Top-Level Module Breakdown ===")
for name, count in sorted(components.items(), key=lambda x: -x[1]):
    pct = 100.0 * count / total if total > 0 else 0
    print("  %-30s  %12s  (%5.1f%%)" % (name, f"{count:,}", pct))
print("  %-30s  %12s" % ("TOTAL", f"{total:,}"))

# Swin detail
print()
print("=== Swin Transformer (RGB Backbone) Stage Detail ===")
if hasattr(model, 'rgb_backbone'):
    stages = {}
    for name, param in model.rgb_backbone.named_parameters():
        parts = name.split('.')
        if parts[0] == 'stages':
            key = 'stages.' + parts[1]
        else:
            key = parts[0]
        stages[key] = stages.get(key, 0) + param.numel()
    for k, v in sorted(stages.items()):
        print("  rgb_backbone.%-35s  %12s" % (k, f"{v:,}"))
    print("  %-35s  %12s" % ("rgb_backbone TOTAL", f"{sum(stages.values()):,}"))

# Depth backbone detail
print()
print("=== Depth Backbone (MobileNetV3) Detail ===")
if hasattr(model, 'depth_backbone'):
    stages = {}
    for name, param in model.depth_backbone.named_parameters():
        parts = name.split('.')
        if parts[0] == 'features':
            if len(parts) > 1 and parts[1].isdigit():
                key = 'features.' + parts[1]
            else:
                key = parts[0]
        else:
            key = parts[0]
        stages[key] = stages.get(key, 0) + param.numel()
    for k, v in sorted(stages.items()):
        print("  depth_backbone.%-35s  %12s" % (k, f"{v:,}"))
    print("  %-35s  %12s" % ("depth_backbone TOTAL", f"{sum(stages.values()):,}"))

# Sem seg head
print()
print("=== Sem Seg Head Detail ===")
if hasattr(model, 'sem_seg_head'):
    sub = {}
    for name, param in model.sem_seg_head.named_parameters():
        parts = name.split('.')
        key = parts[0]
        sub[key] = sub.get(key, 0) + param.numel()
    for k, v in sorted(sub.items()):
        print("  sem_seg_head.%-35s  %12s" % (k, f"{v:,}"))
    print("  %-35s  %12s" % ("sem_seg_head TOTAL", f"{sum(sub.values()):,}"))

    # Pixel decoder deeper
    if hasattr(model.sem_seg_head, 'pixel_decoder'):
        print()
        print("  --- Pixel Decoder Inner ---")
        pd = {}
        for name, param in model.sem_seg_head.pixel_decoder.named_parameters():
            parts = name.split('.')
            key = parts[0]
            pd[key] = pd.get(key, 0) + param.numel()
        for k, v in sorted(pd.items()):
            print("    pixel_decoder.%-35s  %12s" % (k, f"{v:,}"))

# Fusion
print()
print("=== Fusion Module Detail ===")
if hasattr(model, 'fusion_module'):
    sub = {}
    for name, param in model.fusion_module.named_parameters():
        parts = name.split('.')
        key = parts[0]
        sub[key] = sub.get(key, 0) + param.numel()
    for k, v in sorted(sub.items()):
        print("  fusion_module.%-35s  %12s" % (k, f"{v:,}"))
    print("  %-35s  %12s" % ("fusion_module TOTAL", f"{sum(sub.values()):,}"))

# AGPE
print()
print("=== AGPE Module Detail ===")
if hasattr(model, 'agpe'):
    sub = {}
    for name, param in model.agpe.named_parameters():
        parts = name.split('.')
        key = parts[0]
        sub[key] = sub.get(key, 0) + param.numel()
    for k, v in sorted(sub.items()):
        print("  agpe.%-35s  %12s" % (k, f"{v:,}"))
    print("  %-35s  %12s" % ("agpe TOTAL", f"{sum(sub.values()):,}"))

# DPE
print()
print("=== DPE Module Detail ===")
if hasattr(model, 'dpe'):
    sub = {}
    for name, param in model.dpe.named_parameters():
        parts = name.split('.')
        key = parts[0]
        sub[key] = sub.get(key, 0) + param.numel()
    for k, v in sorted(sub.items()):
        print("  dpe.%-35s  %12s" % (k, f"{v:,}"))
    print("  %-35s  %12s" % ("dpe TOTAL", f"{sum(sub.values()):,}"))

# Transformer decoder
print()
print("=== Transformer Decoder Detail ===")
if hasattr(model, 'transformer_decoder'):
    sub = {}
    for name, param in model.transformer_decoder.named_parameters():
        parts = name.split('.')
        key = parts[0]
        sub[key] = sub.get(key, 0) + param.numel()
    for k, v in sorted(sub.items()):
        print("  transformer_decoder.%-35s  %12s" % (k, f"{v:,}"))
    print("  %-35s  %12s" % ("transformer_decoder TOTAL", f"{sum(sub.values()):,}"))

# Summary
print()
print("=" * 70)
print("=== SUMMARY ===")
print("=" * 70)
rgb = components.get('rgb_backbone', 0)
depth = components.get('depth_backbone', 0)
head = components.get('sem_seg_head', 0)
dec = components.get('transformer_decoder', 0)
fusion = components.get('fusion_module', 0)
agpe = components.get('agpe', 0)
dpe = components.get('dpe', 0)
other = total - rgb - depth - head - dec - fusion - agpe - dpe

print("  RGB Backbone (Swin-T):      %12s  (%5.1f%%)" % (f"{rgb:,}", 100*rgb/total))
print("  Depth Backbone (MobileV3):  %12s  (%5.1f%%)" % (f"{depth:,}", 100*depth/total))
print("  Both Backbones:             %12s  (%5.1f%%)" % (f"{rgb+depth:,}", 100*(rgb+depth)/total))
print("  Sem Seg Head (pixel_dec):   %12s  (%5.1f%%)" % (f"{head:,}", 100*head/total))
print("  Transformer Decoder:        %12s  (%5.1f%%)" % (f"{dec:,}", 100*dec/total))
print("  Fusion Module:              %12s  (%5.1f%%)" % (f"{fusion:,}", 100*fusion/total))
print("  AGPE:                       %12s  (%5.1f%%)" % (f"{agpe:,}", 100*agpe/total))
print("  DPE:                        %12s  (%5.1f%%)" % (f"{dpe:,}", 100*dpe/total))
print("  Other:                      %12s  (%5.1f%%)" % (f"{other:,}", 100*other/total))
print("  TOTAL:                      %12s" % f"{total:,}")
