from __future__ import annotations

from pathlib import Path


def test_tools_inference_uses_shared_yolo_overlay() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "inference.py"
    text = script.read_text(encoding="utf-8")
    assert "from magformer.utils.visualization import prediction_to_lists, visualize_predictions" in text
    assert "magformer.visualization.visualizer" not in text
