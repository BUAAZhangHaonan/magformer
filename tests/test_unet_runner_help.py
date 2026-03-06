from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_unet_runner_help_works_as_script() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "baselines" / "run_unet_instance_ecc.py"
    res = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--variant" in res.stdout
