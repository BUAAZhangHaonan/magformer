"""
Data utilities for the vendored UOAIS baseline.

Upstream projects often import dataset mappers at package import time. In this
workspace we want lightweight imports (e.g. unit tests for mask conversion)
to work even when optional deps / compiled ops are not available.
"""

from __future__ import annotations

__all__: list[str] = []

try:
    # Optional convenience re-export.
    from .dataset_mapper import DatasetMapperWithBasis  # noqa: F401

    __all__.append("DatasetMapperWithBasis")
except Exception:
    # DatasetMapperWithBasis may require optional deps (e.g. Perlin noise libs).
    pass

