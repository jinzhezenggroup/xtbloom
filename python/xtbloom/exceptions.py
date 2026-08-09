"""Exception hierarchy for the ``xtbloom`` Python package.

The exception classes mirror the error reporting of comparable tight-binding
Python interfaces (e.g. ``tblite``) while mapping onto the status codes and
per-system diagnostics produced by the xTBloom public C API.
"""

from __future__ import annotations


class XTBloomError(RuntimeError):
    """Base class for all xTBloom Python errors."""


class XTBloomRuntimeError(XTBloomError):
    """An xTBloom C API operation failed at the library level.

    ``status`` carries the raw :c:type:`xtbloom_status_t` value and ``message``
    the library diagnostic from :c:func:`xtbloom_get_last_error`.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class XTBloomValueError(XTBloomError, ValueError):
    """Invalid input was passed to an xTBloom Python object or function."""


class XTBloomNotSupportedError(XTBloomError, NotImplementedError):
    """A requested feature is out of scope for the current build or backend."""
