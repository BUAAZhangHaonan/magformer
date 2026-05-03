from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "analysis" / "prune_checkpoints.py"
    spec = importlib.util.spec_from_file_location("prune_checkpoints", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prune_repaired_unet_keeps_only_best_and_final_weights(tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path / "iaunet"
    out_dir.mkdir()
    for name in [
        "model_best.pth",
        "model_final.pth",
        "model_epoch_0020.pth",
        "trainer_state_latest.pth",
    ]:
        (out_dir / name).write_bytes(b"x")
    (out_dir / "metadata.json").write_text(json.dumps({"model_id": "iaunet"}), encoding="utf-8")

    assert mod._detect_framework(out_dir) == "iaunet"
    removed = mod._prune_repaired_unet(out_dir, dry_run=False)

    assert removed == 2
    assert sorted(path.name for path in out_dir.glob("*.pth")) == ["model_best.pth", "model_final.pth"]


def test_prune_cellpose_removes_nested_official_weight_when_root_final_exists(tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path / "cellpose"
    nested = out_dir / "models"
    nested.mkdir(parents=True)
    (out_dir / "model_final.pth").write_bytes(b"root")
    (nested / "model_final.pth").write_bytes(b"nested")
    (out_dir / "metadata.json").write_text(json.dumps({"model_id": "cellpose"}), encoding="utf-8")

    assert mod._detect_framework(out_dir) == "cellpose"
    removed = mod._prune_repaired_unet(out_dir, dry_run=False)

    assert removed == 1
    assert (out_dir / "model_final.pth").exists()
    assert not (nested / "model_final.pth").exists()
