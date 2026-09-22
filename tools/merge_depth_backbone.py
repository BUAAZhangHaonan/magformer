"""
Merge v314d's depth_backbone weights into v310c / v311c RGB-D checkpoints.
Output new checkpoints that preserve all RGB-D weights except depth_backbone.*
which come from v314d (depth-only trained, AP 0.7058).

Used by experiments v315e (SA-Gate) and v316e (DCCG).
"""
import torch
from pathlib import Path
from copy import deepcopy

BASE_DIR = Path('/home/g203-4028/magformer/output/experiments')

PAIRS = [
    {
        'name': 'v315e (v310c + v314d depth_backbone)',
        'base': BASE_DIR / 'v310c_resume_from_v310a/model_best.pth',
        'output': BASE_DIR / 'v315e_init/model_init.pth',
    },
    {
        'name': 'v316e (v311c + v314d depth_backbone)',
        'base': BASE_DIR / 'v311c_resume_from_v311a/model_best.pth',
        'output': BASE_DIR / 'v316e_init/model_init.pth',
    },
]
DEPTH_SOURCE = BASE_DIR / 'v314d_resume_from_v314b/model_best.pth'


def main():
    # Load v314d depth source once
    print(f"Loading depth source: {DEPTH_SOURCE}")
    depth_ckpt = torch.load(DEPTH_SOURCE, map_location='cpu')
    depth_sd = depth_ckpt['model_state_dict'] if 'model_state_dict' in depth_ckpt else depth_ckpt['model']
    depth_keys_src = {k: v for k, v in depth_sd.items() if k.startswith('depth_backbone.')}
    print(f"  depth_backbone keys in v314d: {len(depth_keys_src)}")

    for pair in PAIRS:
        print(f"\n=== Processing {pair['name']} ===")
        print(f"Loading base: {pair['base']}")
        base_ckpt = torch.load(pair['base'], map_location='cpu')
        # Work on a deep copy so we don't mutate the loaded dict accidentally
        base_sd = deepcopy(base_ckpt['model_state_dict'] if 'model_state_dict' in base_ckpt else base_ckpt['model'])

        # Verify all depth_backbone keys exist in both and shapes match
        base_depth_keys = {k: v for k, v in base_sd.items() if k.startswith('depth_backbone.')}
        print(f"  depth_backbone keys in base: {len(base_depth_keys)}")

        missing_in_base = set(depth_keys_src) - set(base_depth_keys)
        missing_in_src = set(base_depth_keys) - set(depth_keys_src)
        if missing_in_base or missing_in_src:
            raise RuntimeError(f"Key set mismatch! base_only={len(missing_in_src)}, src_only={len(missing_in_base)}")

        shape_mismatches = []
        for k in depth_keys_src:
            if depth_keys_src[k].shape != base_depth_keys[k].shape:
                shape_mismatches.append((k, tuple(base_depth_keys[k].shape), tuple(depth_keys_src[k].shape)))
        if shape_mismatches:
            raise RuntimeError(f"Shape mismatches: {len(shape_mismatches)}\n  first 3: {shape_mismatches[:3]}")

        # Replace
        n_replaced = 0
        for k, v in depth_keys_src.items():
            base_sd[k] = v.clone()
            n_replaced += 1
        print(f"  Replaced {n_replaced} depth_backbone keys with v314d weights")

        # Verify non-depth keys unchanged (compare a sample)
        # (We trust deepcopy + targeted replace; full compare is expensive but let's sample 5)
        import random
        non_depth_keys = [k for k in base_sd if not k.startswith('depth_backbone.')]
        sample = random.sample(non_depth_keys, min(5, len(non_depth_keys)))
        orig_sd = base_ckpt['model_state_dict'] if 'model_state_dict' in base_ckpt else base_ckpt['model']
        for k in sample:
            assert torch.equal(base_sd[k], orig_sd[k]), f"Non-depth key corrupted: {k}"
        print(f"  Verified {len(sample)} non-depth keys unchanged (sample)")

        # Save: preserve base ckpt structure, just replace model_state_dict
        out_ckpt = deepcopy(base_ckpt)
        if 'model_state_dict' in out_ckpt:
            out_ckpt['model_state_dict'] = base_sd
        else:
            out_ckpt['model'] = base_sd
        # Clear iteration/meta fields that might confuse loaders
        out_ckpt['iter'] = 0
        if 'meta' in out_ckpt and isinstance(out_ckpt['meta'], dict):
            out_ckpt['meta']['merged_from'] = str(pair['base']) + ' + ' + str(DEPTH_SOURCE)

        pair['output'].parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_ckpt, pair['output'])
        print(f"  Saved: {pair['output']} ({pair['output'].stat().st_size / 1e6:.1f} MB)")

    print("\n=== All merges complete ===")


if __name__ == '__main__':
    main()
