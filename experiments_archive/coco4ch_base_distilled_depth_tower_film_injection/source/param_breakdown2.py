import sys
sys.path.insert(0, '/home/g203-4028/magformer')
from magformer.config import load_config
from magformer.models.build import build_model

cfg = load_config('/home/g203-4028/magformer/configs/v84_pruned_32k.yaml')
model = build_model(cfg)
model.cuda()
model.eval()

total = sum(p.numel() for p in model.parameters())
print("TOTAL: %s" % f"{total:,}")
print()

# Deep breakdown of each top-level child
for child_name, child_module in model.named_children():
    params = [(n, p) for n, p in child_module.named_parameters()]
    if not params:
        continue

    # Group by first 2-3 levels of nesting
    groups = {}
    for name, param in params:
        parts = name.split('.')
        # Use first 2 parts as key, or first part if only 1
        if len(parts) >= 3:
            key = '.'.join(parts[:3])
        elif len(parts) >= 2:
            key = '.'.join(parts[:2])
        else:
            key = parts[0]
        # But if key is like "model.stages.0", keep it at that level
        # Simplify: group by prefix that makes sense
        groups[key] = groups.get(key, 0) + param.numel()

    child_total = sum(p.numel() for p in child_module.parameters())
    print("=== %s (%s, %.1f%% of total) ===" % (child_name, f"{child_total:,}", 100*child_total/total))

    # Sort by size descending
    for k, v in sorted(groups.items(), key=lambda x: -x[1]):
        pct = 100.0 * v / child_total
        print("  %-55s  %12s  (%5.1f%%)" % (k, f"{v:,}", pct))
    print()

# Now try named_modules to see the full hierarchy
print()
print("=" * 80)
print("=== ALL NAMED MODULES WITH PARAMS (leaf-level) ===")
print("=" * 80)
leaf_modules = []
for name, module in model.named_modules():
    params = sum(p.numel() for p in module.parameters())
    if params > 0 and len(list(module.children())) == 0:  # leaf module
        leaf_modules.append((name, params, type(module).__name__))

# Sort by params descending
leaf_modules.sort(key=lambda x: -x[1])
for name, params, mtype in leaf_modules[:60]:
    print("  %-60s  %12s  %s" % (name, f"{params:,}", mtype))
if len(leaf_modules) > 60:
    print("  ... and %d more leaf modules" % (len(leaf_modules) - 60))
