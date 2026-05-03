from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_ecc_datasets_imports_when_run_from_baselines_dir() -> None:
    """
    Our detectron2 wrappers execute `python baselines/<script>.py`, which makes
    `sys.path[0] == <repo>/baselines`. In that case, imports like
    `from baselines.register_xxx import ...` fail because `baselines` is not a
    package on sys.path.

    This regression test ensures `ecc_datasets.py` can be imported and used
    from that execution context.
    """

    repo_root = Path(__file__).resolve().parents[1]
    baselines_dir = repo_root / "baselines"

    code = (
        "import ecc_datasets; "
        "ecc_datasets.register_ecc_coco('0831', dataset_root='/tmp/does_not_need_to_exist'); "
        "ecc_datasets.register_ecc_coco('0909', dataset_root='/tmp/does_not_need_to_exist'); "
        "print('ok')"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(baselines_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert "ok" in res.stdout

