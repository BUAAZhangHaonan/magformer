from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_write_revisit_comparison_emits_grouped_pairs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "write_revisit_comparison.py"
    out_root = tmp_path / "revisit"
    out_root.mkdir()

    for model_id in [
        "magformer_nodpth_ref",
        "magformer_depthnorm_on",
        "mgm_mask2former_nodpth_ref",
        "mgm_mask2former_depthnorm_on",
    ]:
        model_dir = out_root / model_id
        model_dir.mkdir()
        (model_dir / "metadata.json").write_text(
            json.dumps({"budget": {"iters_per_epoch": 222}}),
            encoding="utf-8",
        )

    summary = {
        "experiment": "0831_1k_20ep_1024_depth_revisit",
        "output_root": str(out_root),
        "magformer_nodpth_ref": {"best": {"iter": 222, "segm": {"AP": 70.0}}, "last": {"iter": 444, "segm": {"AP": 69.0}}},
        "magformer_depthnorm_on": {"best": {"iter": 444, "segm": {"AP": 75.0}}, "last": {"iter": 444, "segm": {"AP": 75.0}}},
        "mgm_mask2former_nodpth_ref": {"best": {"iter": 222, "segm": {"AP": 71.0}}, "last": {"iter": 444, "segm": {"AP": 70.0}}},
        "mgm_mask2former_depthnorm_on": {"best": {"iter": 666, "segm": {"AP": 76.0}}, "last": {"iter": 666, "segm": {"AP": 76.0}}},
    }
    summary_path = out_root / "summary_0831_1k_20ep_1024_depth_revisit.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    subprocess.run(
        [sys.executable, str(script), "--summary", str(summary_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    comparison = json.loads((out_root / "revisit_comparison.json").read_text(encoding="utf-8"))
    assert comparison["pairs"]["magformer"]["control"]["best_ap"] == 70.0
    assert comparison["pairs"]["magformer"]["depth_on"]["best_epoch"] == 2.0
