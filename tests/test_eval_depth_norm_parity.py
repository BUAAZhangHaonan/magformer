from pathlib import Path


def test_evaluate_and_export_explicitly_forward_depth_per_sample_norm():
    repo_root = Path(__file__).resolve().parents[1]
    evaluate_py = (repo_root / "tools" / "evaluate.py").read_text(encoding="utf-8")
    export_py = (repo_root / "tools" / "export_results.py").read_text(encoding="utf-8")

    assert "depth_per_sample_norm=getattr(data_cfg.depth, \"per_sample_norm\", True)" in evaluate_py
    assert "depth_per_sample_norm=getattr(config.data.depth, \"per_sample_norm\", True)" in export_py
