from __future__ import annotations

from pathlib import Path


def test_lightdepth_stage_b_all_lists_expected_candidates() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_lightdepth_stage_b_all.sh"
    text = script.read_text(encoding="utf-8")
    assert "magformer_lightdepth_mobilenetv3_directadd_edge" in text
    assert "magformer_nodpth_ref" in text
    assert "mobilenetv3_crossattn_edge" in text
    assert "mobilenetv3_crossattn_edge_validhole" in text
    assert "mobilenetv3_crossattn_edge_validhole_variance" in text
