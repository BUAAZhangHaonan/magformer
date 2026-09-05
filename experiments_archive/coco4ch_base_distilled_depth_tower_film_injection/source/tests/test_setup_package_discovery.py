from __future__ import annotations

import runpy
from unittest.mock import patch
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_setup_package_discovery_excludes_duplicate_packages() -> None:
    with patch("setuptools.setup") as setup_mock:
        runpy.run_path(str(REPO_ROOT / "setup.py"), run_name="__main__")

    packages = set(setup_mock.call_args.kwargs["packages"])

    assert {"magformer", "magformer.engine", "magformer.models", "tools"} <= packages
    assert "engine" not in packages
    assert "models" not in packages
    assert all(
        package == "magformer"
        or package.startswith("magformer.")
        or package == "tools"
        or package.startswith("tools.")
        for package in packages
    )
