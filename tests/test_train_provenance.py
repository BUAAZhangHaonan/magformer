from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def _config_dump(output_dir: Path) -> dict:
    return {
        "name": "vc_suda_stage_b_1024_teacher8499",
        "runtime": {"output_dir": str(output_dir), "seed": 123},
        "vc_suda": {"enabled": True, "stage": "B"},
    }


def test_write_run_provenance_writes_config_snapshot_and_metadata(tmp_path, monkeypatch):
    from tools import train as train_tool

    config = SimpleNamespace(model_dump=lambda: _config_dump(tmp_path))
    monkeypatch.setattr(
        train_tool,
        "_collect_git_metadata",
        lambda repo_root: {
            "commit": "abc123",
            "branch": "feature/test",
            "status_short": [" M tools/train.py"],
            "dirty": True,
        },
    )

    latest_path = train_tool.write_run_provenance(
        config=config,
        output_dir=tmp_path,
        argv=["tools/train.py", "--config", "configs/stage_b.yaml", "--eval-only"],
        config_path="configs/stage_b.yaml",
        eval_only=True,
        repo_root=Path("/repo"),
        update_latest=True,
    )

    resolved = yaml.safe_load((tmp_path / "config_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved == _config_dump(tmp_path)

    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["argv"] == ["tools/train.py", "--config", "configs/stage_b.yaml", "--eval-only"]
    assert metadata["command"] == "tools/train.py --config configs/stage_b.yaml --eval-only"
    assert metadata["config_path"] == "configs/stage_b.yaml"
    assert metadata["eval_only"] is True
    assert metadata["git"]["commit"] == "abc123"
    assert metadata["git"]["dirty"] is True
    assert isinstance(metadata["timestamp_utc"], str)
    assert Path(metadata["latest_symlink"]) == latest_path
    assert latest_path.name == "vc_suda_stage_b_1024_teacher8499_latest"
    assert latest_path.is_symlink()
    assert latest_path.resolve() == tmp_path.resolve()


def test_update_stage_latest_symlink_replaces_only_existing_symlink(tmp_path):
    from tools import train as train_tool

    first = tmp_path / "vc_suda_stage_b_1024_teacher8499_20260514_0100"
    second = tmp_path / "vc_suda_stage_b_1024_teacher8499_20260514_0200"
    first.mkdir()
    second.mkdir()

    latest = train_tool.update_stage_latest_symlink(
        first,
        latest_name="vc_suda_stage_b_1024_teacher8499_latest",
    )
    assert latest.is_symlink()
    assert latest.resolve() == first.resolve()

    same_latest = train_tool.update_stage_latest_symlink(
        second,
        latest_name="vc_suda_stage_b_1024_teacher8499_latest",
    )
    assert same_latest == latest
    assert latest.is_symlink()
    assert latest.resolve() == second.resolve()
    assert first.exists()


def test_update_stage_latest_symlink_refuses_non_symlink_path(tmp_path):
    from tools import train as train_tool

    run_dir = tmp_path / "vc_suda_stage_b_1024_teacher8499_20260514_0300"
    run_dir.mkdir()
    occupied = tmp_path / "vc_suda_stage_b_1024_teacher8499_latest"
    occupied.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to replace non-symlink latest path"):
        train_tool.update_stage_latest_symlink(
            run_dir,
            latest_name="vc_suda_stage_b_1024_teacher8499_latest",
        )

    assert occupied.is_dir()


def test_stage_latest_name_prefers_config_name():
    from tools import train as train_tool

    config = SimpleNamespace(
        name="vc_suda_stage_b_1024_teacher8499",
        vc_suda=SimpleNamespace(enabled=True, stage="B"),
    )

    assert train_tool.stage_latest_name(config) == "vc_suda_stage_b_1024_teacher8499_latest"


def test_stage_latest_name_falls_back_to_vc_suda_stage():
    from tools import train as train_tool
