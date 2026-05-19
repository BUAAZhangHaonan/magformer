from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "tools" / "plan_checkpoint_cleanup.py"
    spec = importlib.util.spec_from_file_location("plan_checkpoint_cleanup", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch(path: Path, size: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_referenced_checkpoint_is_not_cleanup_candidate(tmp_path: Path) -> None:
    mod = _load_module()
    referenced = tmp_path / "output" / "run_a" / "checkpoint_iter_0000100.pth"
    unreferenced = tmp_path / "output" / "run_a" / "checkpoint_iter_0000200.pth"
    _touch(referenced, size=10)
    _touch(unreferenced, size=20)
    note = tmp_path / "docs" / "results" / "note.md"
    note.parent.mkdir(parents=True)
    note.write_text(f"keep output/run_a/{referenced.name}\n", encoding="utf-8")

    plan = mod.build_cleanup_plan(tmp_path)
    candidates = {item["path"] for item in plan["delete_candidates"]}
    keep = {item["path"] for item in plan["kept_checkpoints"]}

    assert "output/run_a/checkpoint_iter_0000100.pth" in keep
    assert "output/run_a/checkpoint_iter_0000100.pth" not in candidates
    assert "output/run_a/checkpoint_iter_0000200.pth" in candidates


def test_final_gate_keep_list_is_applied(tmp_path: Path) -> None:
    mod = _load_module()
    keep_paths = [
        "output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0002000.pth",
        "output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0002000.pth",
        "output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0000999.pth",
        "output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0001499.pth",
        "output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0002000.pth",
        "output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0000999.pth",
        "output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0002000.pth",
        "output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0000499.pth",
        "output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0000799.pth",
        "output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0001000.pth",
        "output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0000499.pth",
        "output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0000799.pth",
        "output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0001000.pth",
        "output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0001499.pth",
        "output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0002000.pth",
        "output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth",
        "output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth",
    ]
    for rel_path in keep_paths:
        _touch(tmp_path / rel_path)
    _touch(
        tmp_path
        / "output"
        / "baseline"
        / "r139_magformer_rgb_only_r98warm_target150_20260519"
        / "checkpoint_iter_0000199.pth"
    )

    plan = mod.build_cleanup_plan(tmp_path)
    candidates = {item["path"] for item in plan["delete_candidates"]}
    keep = {item["path"] for item in plan["kept_checkpoints"]}

    assert set(keep_paths).issubset(keep)
    assert not set(keep_paths) & candidates
    assert (
        "output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000199.pth"
        in candidates
    )


def test_planner_does_not_delete_files_and_writes_manifest(tmp_path: Path) -> None:
    mod = _load_module()
    candidate = tmp_path / "output" / "run_b" / "checkpoint_iter_0000300.pth"
    log_file = tmp_path / "output" / "run_b" / "metrics.json"
    _touch(candidate, size=11)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "docs" / "results" / "checkpoint_cleanup_manifest_20260519.md"

    exit_code = mod.main(["--repo-root", str(tmp_path), "--manifest", str(manifest)])

    assert exit_code == 0
    assert candidate.exists()
    assert log_file.exists()
    text = manifest.read_text(encoding="utf-8")
    assert "output/run_b/checkpoint_iter_0000300.pth" in text
    assert "metrics.json" not in text

    rerun_plan = mod.build_cleanup_plan(tmp_path)
    rerun_candidates = {item["path"] for item in rerun_plan["delete_candidates"]}

    assert "output/run_b/checkpoint_iter_0000300.pth" in rerun_candidates
