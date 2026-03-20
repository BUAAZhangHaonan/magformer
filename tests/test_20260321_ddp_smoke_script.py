from __future__ import annotations

import subprocess
from pathlib import Path


def test_ddp_smoke_dry_run_contains_expected_multi_family_commands(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "run_20260321_ddp_smoke_canary.sh"

    res = subprocess.run(
        [
            "bash",
            str(script),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "torchrun --nproc_per_node=2 tools/train.py" in res.stdout
    assert "train_net_mgm_0831.py --num-gpus 2" in res.stdout
    assert "run_uoais_ecc.py" in res.stdout
    assert "yolo segment train" in res.stdout
    assert "device=0,1" in res.stdout
    assert "run_unet_instance_ecc.py" in res.stdout

