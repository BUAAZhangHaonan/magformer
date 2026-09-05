from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_lightdepth_roster_keeps_top3_and_archives_others() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    roster_path = repo_root / "configs" / "experiments" / "lightdepth_roster.json"
    payload = json.loads(roster_path.read_text(encoding="utf-8"))

    active = payload["active"]
    archived = payload["archived"]

    assert [item["variant"] for item in active] == [
        "convnextlite_spatialgate_edge_validhole",
        "mobilenetv3_sagate_edge_validhole",
        "mobilenetv3_spatialgate_edge_validhole",
    ]
    assert len(active) == 3
    assert len(archived) == 19
    assert all(item["status"] == "active" for item in active)
    assert all(item["status"] == "archived" for item in archived)


def test_lightdepth_roster_cli_lists_active_variants() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "lightdepth_roster.py"

    res = subprocess.run(
        [
            "python",
            str(script),
            "--status",
            "active",
            "--field",
            "variant",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )

    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    assert lines == [
        "convnextlite_spatialgate_edge_validhole",
        "mobilenetv3_sagate_edge_validhole",
        "mobilenetv3_spatialgate_edge_validhole",
    ]
