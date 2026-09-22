from __future__ import annotations

import os
import subprocess
import sys
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


def test_mgm_dataset_package_exists() -> None:
    base = (
        Path(__file__).resolve().parents[1]
        / "baselines"
        / "MGM_Mask2Former"
        / "mask2former"
        / "data"
        / "datasets"
    )
    assert (base / "__init__.py").is_file()
    assert (base / "register_coco_rgbd_instance.py").is_file()


def test_mgm_dataset_importable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline_root = repo_root / "baselines" / "MGM_Mask2Former"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{baseline_root}:{existing_pythonpath}" if existing_pythonpath else str(baseline_root)
    )
    code = (
        "import importlib; "
        "module = importlib.import_module('mask2former.data.datasets.register_coco_rgbd_instance'); "
        "assert hasattr(module, 'register_all_coco_rgbd')"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
