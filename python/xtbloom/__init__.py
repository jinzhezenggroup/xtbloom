"""xTBloom: batched GFN-xTB energy and analytic-force inference.

The Python package wraps the public xTBloom C ABI through :mod:`ctypes`,
marshals numpy arrays between Python and the C descriptors, and exposes a
tblite-like high-level interface (:class:`xtbloom.interface.Calculator`),
the native ragged-batch model (:class:`xtbloom.interface.BatchCalculator`),
plus optional ASE (:mod:`xtbloom.ase`) and dpdata (:mod:`xtbloom.dpdata`)
integrations.

Atomic units everywhere: bohr, Hartree, Hartree/bohr, elementary charge.
"""

from importlib.metadata import version as _distribution_version

from ._dlpack import DLPackResultBuffer
from .exceptions import (
    XTBloomError,
    XTBloomNotSupportedError,
    XTBloomRuntimeError,
    XTBloomValueError,
)
from .interface import (
    ArrayBatch,
    ArrayBatchResult,
    BatchCalculator,
    BatchResult,
    Calculator,
    ChargeResponse,
    Context,
    PointCharge,
    Result,
    Structure,
    compute_arrays,
    numbers_to_symbols,
    symbols_to_numbers,
)
from .torch import xtbloom_torch

# The installed distribution metadata is resolved from Git tags by the build
# backend. Reading it here avoids a second generated or hand-maintained Python
# version source.
__version__ = _distribution_version("xtbloom")
del _distribution_version

__all__ = [
    "ArrayBatch",
    "ArrayBatchResult",
    "BatchCalculator",
    "BatchResult",
    "Calculator",
    "ChargeResponse",
    "Context",
    "DLPackResultBuffer",
    "PointCharge",
    "Result",
    "Structure",
    "XTBloomError",
    "XTBloomNotSupportedError",
    "XTBloomRuntimeError",
    "XTBloomValueError",
    "__version__",
    "compute_arrays",
    "numbers_to_symbols",
    "symbols_to_numbers",
    "xtbloom_torch",
]
