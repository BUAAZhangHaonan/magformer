from __future__ import annotations

from pathlib import Path


def test_msmformer_meanshiftformer_importable() -> None:
    """
    The vendored MSMFormer baseline is pruned to "model code only".
    We still want the core `meanshiftformer` package to be importable so that
    Detectron2 registries are populated (meta-arch, heads, pixel decoders).
    """
    repo_root = Path(__file__).resolve().parents[1]
    msmformer_root = repo_root / "baselines" / "icra_2026" / "msmformer" / "MSMFormer"
    assert msmformer_root.exists()

    import sys

    sys.path.insert(0, str(msmformer_root))

    import meanshiftformer  # noqa: F401

    assert hasattr(meanshiftformer, "PretrainedMeanShiftMaskFormer")

