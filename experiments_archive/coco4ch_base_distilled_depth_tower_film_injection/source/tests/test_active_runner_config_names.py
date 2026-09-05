from __future__ import annotations

from pathlib import Path


def test_active_msmformer_uoais_runners_use_tracks_config_names() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scripts = [
        repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_msmformer.sh",
        repo_root / "scripts" / "experiments" / "run_ecc_20ep_trackp_msmformer.sh",
        repo_root / "scripts" / "experiments" / "run_ecc_20ep_tracks_msmformer.sh",
        repo_root / "scripts" / "experiments" / "run_0831_1k_20ep_scratch_uoais.sh",
        repo_root / "scripts" / "experiments" / "run_ecc_20ep_trackp_uoais.sh",
        repo_root / "scripts" / "experiments" / "run_ecc_20ep_tracks_uoais.sh",
    ]
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "_5k_scratch" not in text
        assert "_tracks.yaml" in text
