"""
AdelaiDet / UOAIS (vendored baseline).

Upstream `adet` typically imports `modeling` at package import time to register
custom components into Detectron2 registries.

However, `adet.modeling` imports custom CUDA/C++ ops (`adet._C`). For unit tests
and lightweight utilities (e.g. dataset mappers) we want `import adet` to work
even when these ops are not built yet.
"""

__version__ = "0.1.1"

try:
    # Import for side effects: registers custom components for training.
    from adet import modeling  # noqa: F401
except Exception:
    # Allow importing dataset/utility modules without compiled ops.
    pass
