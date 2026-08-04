"""Exception hierarchy for the ``gpuxtb`` Python package.

The exception classes mirror the error reporting of comparable tight-binding
Python interfaces (e.g. ``tblite``) while mapping onto the status codes and
per-system diagnostics produced by the gpuxtb public C API.
"""

from __future__ import annotations

from typing import Optional


class GPUxtbError(RuntimeError):
    """Base class for all gpuxtb Python errors."""


class GPUxtbRuntimeError(GPUxtbError):
    """A gpuxtb C API operation failed at the library level.

    ``status`` carries the raw :c:type:`gpuxtb_status_t` value and ``message``
    the library diagnostic from :c:func:`gpuxtb_get_last_error`.
    """

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class GPUxtbValueError(GPUxtbError, ValueError):
    """Invalid input was passed to a gpuxtb Python object or function."""


class GPUxtbNotSupportedError(GPUxtbError, NotImplementedError):
    """A requested feature is out of scope for the current build or backend."""
