from __future__ import annotations

from pathlib import Path


def test_trackp_yolo_runner_uses_managed_pretrained_weight_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_ecc_20ep_trackp_yolov8_seg.sh"
    text = script.read_text(encoding="utf-8")
    assert "output/pretrained/yolov8${MODEL_SIZE}-seg.pt" in text
    assert "model=yolov8n-seg.pt" not in text


def test_0831_yolo_runner_supports_dynamic_model_size_weight_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_yolov8_seg.sh"
    text = script.read_text(encoding="utf-8")
    assert "MODEL_SIZE" in text
    assert "yolov8${MODEL_SIZE}-seg.yaml" in text
    assert "yolov8${MODEL_SIZE}-seg.pt" in text
