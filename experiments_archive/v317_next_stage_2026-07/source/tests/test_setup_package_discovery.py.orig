from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_find_packages_discovers_magformer_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools import find_packages; print(find_packages())",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "magformer" in result.stdout
