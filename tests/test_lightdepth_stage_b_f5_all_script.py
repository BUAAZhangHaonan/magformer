from __future__ import annotations

import subprocess
from pathlib import Path


def test_lightdepth_stage_b_f5_all_lists_expected_candidates_and_anchors() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_lightdepth_stage_b_f5_all.sh"
    text = script.read_text(encoding="utf-8")
    assert "mobilenetv3_priorguidedcrossattn_edge" in text
    assert "mobilenetv3_priorguidedcrossattn_edge_validhole" in text
    assert "mobilenetv3_priorguidedcrossattn_edge_validhole_variance" in text
    assert "convnextlite_priorguidedcrossattn_edge" in text
    assert "convnextlite_priorguidedcrossattn_edge_validhole" in text
    assert "convnextlite_priorguidedcrossattn_edge_validhole_variance" in text
    assert "magformer_nodpth_ref" in text
    assert "magformer_lightdepth_convnextlite_spatialgate_edge_validhole" in text
    assert "magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole" in text


def test_lightdepth_stage_b_f5_all_skips_completed_candidates_in_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_lightdepth_stage_b_f5_all.sh"
    output_root = tmp_path / "out"
    done_dir = output_root / "magformer_lightdepth_mobilenetv3_priorguidedcrossattn_edge"
    done_dir.mkdir(parents=True)
    (done_dir / "metrics.cocoeval.json").write_text("{}", encoding="utf-8")

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "SKIP mobilenetv3_priorguidedcrossattn_edge" in res.stdout
