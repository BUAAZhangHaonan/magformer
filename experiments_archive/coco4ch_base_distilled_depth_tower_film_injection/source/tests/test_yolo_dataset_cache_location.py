from __future__ import annotations

from pathlib import Path


def test_active_yolo_runners_no_longer_use_output_baselines() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scripts = [
        repo_root / "scripts" / "experiments" / "run_ecc_20ep_trackp_yolov8_seg.sh",
        repo_root / "scripts" / "experiments" / "run_ecc_20ep_tracks_yolov8_seg.sh",
        repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_yolov8_seg.sh",
    ]
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "output/baselines" not in text
        assert "/_shared/yolo_" in text


def test_1024_revisit_all_excludes_msmformer() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_depth_revisit_all.sh"
    text = script.read_text(encoding="utf-8")
    assert "msmformer_1024" not in text


def test_1024_revisit_all_excludes_ucn() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_depth_revisit_all.sh"
    text = script.read_text(encoding="utf-8")
    assert "ucn_1024" not in text
