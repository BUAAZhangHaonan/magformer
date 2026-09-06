from __future__ import annotations

from pathlib import Path


def test_extended_metrics_table_default_summaries_include_f5_suite() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "write_extended_metrics_table.py"
    text = script.read_text(encoding="utf-8")
    assert (
        "output/experiments/0831_1k_20ep_1024_lightdepth_stage_b_f5/"
        "summary_0831_1k_20ep_1024_lightdepth_stage_b_f5.json"
    ) in text
