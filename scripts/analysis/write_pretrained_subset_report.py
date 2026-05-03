#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


PRETRAINED_KEYS = [
    "magformer_depthnorm_on",
    "mgm_mask2former_depthnorm_on",
    "official_mask2former_pretrained",
    "maskrcnn_pretrained",
    "yolov8_seg_pretrained",
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_ap(summary: Dict[str, Any], key: str) -> float | None:
    entry = summary.get(key, {})
    return ((entry.get("best") or {}).get("segm") or {}).get("AP")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    summary = _load_json(Path(args.summary))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Pretrained Subset Report",
        "",
        f"- Source summary: `{Path(args.summary)}`",
        "",
        "## Ranking",
        "",
        "| Model | Best segm AP |",
        "|---|---:|",
    ]
    rows = []
    for key in PRETRAINED_KEYS:
        ap_value = _best_ap(summary, key)
        rows.append((ap_value if ap_value is not None else -1.0, key, ap_value))
    for _, key, ap_value in sorted(rows, reverse=True):
        lines.append(f"| {key} | {ap_value:.3f} |")

    nodpth = _best_ap(summary, "mgm_mask2former_nodpth_ref")
    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- `mgm_mask2former_nodpth_ref` is included as a diagnostic reference only.",
        "- It is **not directly comparable** to `mgm_mask2former_depthnorm_on` as a symmetric no-depth ablation, because it is initialized from a generic public COCO checkpoint with depth/MGM/DPE disabled rather than from a dedicated 0831 no-depth reference model.",
    ]
    if nodpth is not None:
        lines.append(f"- Current `mgm_mask2former_nodpth_ref` best AP: `{nodpth:.3f}`.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[pretrained-subset-report] wrote: {out_path}")


if __name__ == "__main__":
    main()
