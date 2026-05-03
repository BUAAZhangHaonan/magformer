from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

RECOMMENDED_CONFIGS = [
    REPO_ROOT / "configs" / "magformer_0831_1k_60ep_trackp.yaml",
    REPO_ROOT / "configs" / "magformer_0831_1k_mgm_finetune.yaml",
    REPO_ROOT / "configs" / "magformer_track_b_2k.yaml",
    REPO_ROOT / "configs" / "magformer_track_b_20k.yaml",
]

OWNED_DOCS = [
    REPO_ROOT / "docs" / "archive" / "2026-04-12-pre-remediation" / "experiments" / "track_a_2k_report.md",
    REPO_ROOT / "docs" / "archive" / "2026-04-12-pre-remediation" / "experiments" / "track_b_2k_report.md",
    REPO_ROOT / "docs" / "archive" / "2026-04-12-pre-remediation" / "experiments" / "baselines_0831_1k_20ep_scratch8_report.md",
    REPO_ROOT / "docs" / "archive" / "2026-04-12-pre-remediation" / "experiments" / "baselines_0831_1k_5k_scratch8_report.md",
]

TRACK_RUNNERS = [
    REPO_ROOT / "scripts" / "experiments" / "run_track_b_2k.sh",
    REPO_ROOT / "scripts" / "experiments" / "run_track_b_20k.sh",
]


def test_recommended_configs_do_not_ship_with_warm_start_paths() -> None:
    for path in RECOMMENDED_CONFIGS:
        text = path.read_text(encoding="utf-8")
        assert "finetune_weights: null" in text
        assert "/home/" not in text


def test_owned_docs_are_portable_and_mark_draft_when_incomplete() -> None:
    for path in OWNED_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text
        if "TBD" in text:
            assert re.search(r"(?i)\bdraft\b", text), path


def test_track_b_runners_render_explicit_magformer_weight_override(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"

    for script in TRACK_RUNNERS:
        result = subprocess.run(
            [
                "bash",
                str(script),
                "--dry-run",
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(output_root / script.stem),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )

        assert "tools/train.py" in result.stdout
        assert "--finetune-weights" in result.stdout
        assert "track_a_2k/magformer/model_best.pth" in result.stdout


def test_requirements_dev_includes_pytest() -> None:
    text = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert re.search(r"(?m)^pytest\b", text)
