"""Backend-neutral array probing built on the ``array_api_compat`` shims.

gpuxtb consumes arrays only through the DLPack producer protocol (see
:mod:`gpuxtb._dlpack`).  This module uses ``array_api_compat`` -- a small,
backend-neutral, MIT-licensed helper with no runtime dependencies -- to name
the caller's backend, detect lazy/tracer objects, and answer the writeability
question that the raw capsule format cannot always express.  No CuPy, JAX, or
PyTorch package is imported by gpuxtb runtime code: the shims import a
producer module lazily only after the caller already supplied one of that
backend's arrays.

Nothing here evaluates or copies array data; probing is metadata-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .exceptions import GPUxtbValueError

if TYPE_CHECKING:
    from types import ModuleType


def _compat() -> ModuleType:
    """Import the ``array_api_compat`` shim package (a declared dependency)."""
    try:
        import array_api_compat as compat
    except ImportError:  # pragma: no cover - packaging failure, not user error
        raise GPUxtbValueError(
            "gpuxtb requires the array_api_compat package to consume Array API arrays"
        ) from None
    return compat


def backend_name(array: object) -> str:
    """Return a stable human-readable backend name for one array."""
    if isinstance(array, np.ndarray):
        return "NumPy"
    compat = _compat()
    if compat.is_torch_array(array):
        return "PyTorch"
    if compat.is_cupy_array(array):
        return "CuPy"
    if compat.is_jax_array(array):
        return "JAX"
    return type(array).__name__


def is_lazy(array: object) -> bool:
    """Return whether the array is a framework tracer or graph node.

    The probe inspects only metadata (never evaluates the object).  JAX and
    PyTorch tracers are rejected by the DLPack protocol itself, but a precise
    early diagnostic is friendlier than an obscure ``__dlpack__`` failure.
    """
    compat = _compat()
    if not compat.is_torch_array(array) and not compat.is_jax_array(array):
        return False
    return (
        bool(getattr(array, "is_tracer", False))
        or getattr(array, "grad_fn", None) is not None
    )


def is_writable(array: object) -> bool | None:
    """Return a trustworthy mutability hint, or ``None`` when unavailable.

    ``array_api_compat.is_writeable_array`` intentionally assumes unknown
    Array API objects are writable.  That is useful for generic dispatch but
    is not strong enough to authorize native writes through a legacy DLPack
    capsule, which has no read-only flag of its own.
    """
    if isinstance(array, np.ndarray):
        return bool(array.flags.writeable)
    compat = _compat()
    if compat.is_jax_array(array):
        return False
    if compat.is_torch_array(array) or compat.is_cupy_array(array):
        return bool(compat.is_writeable_array(array))
    return None
