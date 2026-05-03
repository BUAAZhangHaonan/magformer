from __future__ import annotations

from pathlib import Path


def test_lightdepth_stage_a_all_lists_expected_candidates() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_lightdepth_stage_a_all.sh"
    text = script.read_text(encoding="utf-8")
    assert "magformer_nodpth_ref" in text
    assert "lightdepth_roster.py" in text
    assert "--status active --field variant" in text
    assert "mobilenetv3_directadd_edge" not in text
