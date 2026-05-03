from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.benchmark_inference import _detect_register_id


def test_detect_register_id_prefers_metadata_register(tmp_path: Path) -> None:
    out_dir = tmp_path / "model"
    out_dir.mkdir(parents=True)
    (out_dir / "metadata.json").write_text(json.dumps({"register": "20260318_1K_1566"}), encoding="utf-8")

    register = _detect_register_id(out_dir, tmp_path / "fallback_name")
    assert register == "20260318_1K_1566"


def test_detect_register_id_falls_back_to_dataset_root_name(tmp_path: Path) -> None:
    out_dir = tmp_path / "model"
    out_dir.mkdir(parents=True)
    dataset_root = tmp_path / "20260318_1K_1566"
    dataset_root.mkdir(parents=True)

    register = _detect_register_id(out_dir, dataset_root)
    assert register == "20260318_1K_1566"
