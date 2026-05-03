from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "baselines" / "baseline_fidelity.py"
    spec = importlib.util.spec_from_file_location("baseline_fidelity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_unet_baseline_fidelity_registry_is_explicit() -> None:
    mod = _load_module()

    cellpose = mod.baseline_fidelity_for("cellpose")
    iaunet = mod.baseline_fidelity_for("iaunet")
    stardist = mod.baseline_fidelity_for("stardist")

    assert cellpose["implementation_kind"] == "official-library"
    assert cellpose["official_code_used"] is True
    assert cellpose["paper_faithful"] is True
    assert "cellpose" in cellpose["academic_claim"].lower()

    assert iaunet["implementation_kind"] == "paper-faithful-reimplementation"
    assert iaunet["official_code_used"] is False
    assert iaunet["paper_faithful"] is True
    assert "maskness" in " ".join(iaunet["known_limitations"]).lower()

    assert stardist["implementation_kind"] == "official-library"
    assert stardist["official_code_used"] is True
    assert not stardist["known_limitations"]


def test_iaunet_architecture_audit_records_paper_contract() -> None:
    mod = _load_module()
    audit = mod.audit_iaunet_defaults()

    failed = {check["name"] for check in audit["checks"] if not check["passed"]}
    assert failed == set()
    assert audit["paper_faithful"] is True
