from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "check_r121_r122_resume_state.py"
HELPER = REPO_ROOT / "tools" / "r122_formal_eval_paths.py"
R122_DIR = Path("output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300")


def _load_helper():
    spec = importlib.util.spec_from_file_location("r122_formal_eval_paths", HELPER)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(repo_root: Path, relative: Path, text: str = "ok") -> Path:
    path = repo_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _r121_passed(repo_root: Path) -> None:
    _write(
        repo_root,
        Path("output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075/train.log"),
        "depth sanity passed\niter: 0/1 loss_depth_boundary: 0.12\nmax_iter: 1\ntotal training time\n",
    )


def test_final_label_selects_latest_r122_checkpoint_and_final_artifacts(tmp_path: Path) -> None:
    helper = _load_helper()
    for iteration in (99, 199, 300):
        _write(tmp_path, R122_DIR / f"checkpoint_iter_{iteration:07d}.pth")

    spec = helper.resolve_r122_formal_eval_artifacts(tmp_path, checkpoint_label="final")

    assert spec.checkpoint == tmp_path / R122_DIR / "checkpoint_iter_0000300.pth"
    assert spec.label == "final"
    assert spec.remaining75_dir == Path(
        "output/diagnostics/r122_depth_boundary_w001_final_remaining75_1024_backmap_topk200_20260518"
    )
    assert spec.val28_dir == Path(
        "output/diagnostics/r122_depth_boundary_w001_final_val28_1024_backmap_topk200_20260518"
    )
    assert spec.bucket_compare_dir == Path(
        "output/diagnostics/r122_depth_boundary_w001_final_bucket_compare_20260518"
    )
    assert spec.go_no_go_json == Path("output/diagnostics/r122_depth_boundary_w001_final_go_no_go_20260518/go_no_go.json")


def test_iter0099_label_preserves_existing_r122_artifact_paths(tmp_path: Path) -> None:
    helper = _load_helper()
    _write(tmp_path, R122_DIR / "checkpoint_iter_0000099.pth")

    spec = helper.resolve_r122_formal_eval_artifacts(tmp_path, checkpoint_label="iter0099")

    assert spec.checkpoint == tmp_path / R122_DIR / "checkpoint_iter_0000099.pth"
    assert spec.remaining75_dir == Path(
        "output/diagnostics/r122_depth_boundary_w001_iter0099_remaining75_1024_backmap_topk200_20260518"
    )
    assert spec.val28_dir == Path(
        "output/diagnostics/r122_depth_boundary_w001_iter0099_val28_1024_backmap_topk200_20260518"
    )


def test_unknown_r122_checkpoint_label_fails_loudly(tmp_path: Path) -> None:
    helper = _load_helper()

    try:
        helper.resolve_r122_formal_eval_artifacts(tmp_path, checkpoint_label="iter42")
    except helper.R122CheckpointSelectionError as exc:
        assert "unknown R122 checkpoint label" in str(exc)
        assert "iter0099" in str(exc)
        assert "final" in str(exc)
    else:
        raise AssertionError("unknown R122 checkpoint label did not fail")


def test_explicit_missing_r122_checkpoint_path_fails_loudly(tmp_path: Path) -> None:
    helper = _load_helper()

    missing = tmp_path / R122_DIR / "checkpoint_iter_0000300.pth"
    try:
        helper.resolve_r122_formal_eval_artifacts(tmp_path, checkpoint_label="final", checkpoint_path=missing)
    except helper.R122CheckpointSelectionError as exc:
        assert "missing requested R122 checkpoint" in str(exc)
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing explicit R122 checkpoint did not fail")


def test_resume_state_checker_uses_requested_final_checkpoint_and_artifact_paths(tmp_path: Path) -> None:
    _r121_passed(tmp_path)
    _write(tmp_path, R122_DIR / "checkpoint_iter_0000099.pth")
    _write(tmp_path, R122_DIR / "checkpoint_iter_0000300.pth")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(tmp_path), "--r122-checkpoint-label", "final"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "checkpoint_iter_0000300.pth" in result.stdout
    assert "r122_depth_boundary_w001_final_remaining75_1024_backmap_topk200_20260518" in result.stdout
    assert "checkpoint_iter_0000099.pth" not in result.stdout


def test_resume_state_checker_errors_on_unknown_r122_checkpoint_label(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(tmp_path), "--r122-checkpoint-label", "not-a-label"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unknown R122 checkpoint label" in result.stderr
