from __future__ import annotations

from pathlib import Path


def test_active_mgm_scripts_use_repo_local_baseline_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scripts = [
        repo_root / "scripts" / "experiments" / "run_ecc_20ep_trackp_mgm_mask2former.sh",
        repo_root / "scripts" / "experiments" / "run_ecc_20ep_tracks_mgm_mask2former.sh",
        repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_mgm_mask2former.sh",
        repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh",
    ]
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert '${REPO_ROOT}/baselines/MGM_Mask2Former' in text
        assert '${PROJECT_ROOT}/mask2former/MGM_Mask2Former' not in text


def test_mgm_depth_control_tests_read_repo_local_sources() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    test_text = (repo_root / "tests" / "test_mgm_depth_controls.py").read_text(encoding="utf-8")
    assert ' / "baselines"' in test_text
    assert ' / "MGM_Mask2Former"' in test_text
    assert '/../mask2former/MGM_Mask2Former/' not in test_text
