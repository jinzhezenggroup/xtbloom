"""gpuxtb: batched GFN-xTB energy and analytic-force inference.

The Python package wraps the public gpuxtb C ABI through :mod:`ctypes`,
marshals numpy arrays between Python and the C descriptors, and exposes a
tblite-like high-level interface (:class:`gpuxtb.interface.Calculator`),
the native ragged-batch model (:class:`gpuxtb.interface.BatchCalculator`),
plus optional ASE (:mod:`gpuxtb.ase`) and dpdata (:mod:`gpuxtb.dpdata`)
integrations.

Atomic units everywhere: bohr, Hartree, Hartree/bohr, elementary charge.
"""

__version__ = "0.1.0"

from ._dlpack import DLPackResultBuffer
from .exceptions import (
    GPUxtbError,
    GPUxtbNotSupportedError,
    GPUxtbRuntimeError,
    GPUxtbValueError,
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
from .torch import gpuxtb_torch

__all__ = [
    "ArrayBatch",
    "ArrayBatchResult",
    "BatchCalculator",
    "BatchResult",
    "Calculator",
    "ChargeResponse",
    "Context",
    "DLPackResultBuffer",
    "GPUxtbError",
    "GPUxtbNotSupportedError",
    "GPUxtbRuntimeError",
    "GPUxtbValueError",
    "PointCharge",
    "Result",
    "Structure",
    "__version__",
    "compute_arrays",
    "gpuxtb_torch",
    "numbers_to_symbols",
    "symbols_to_numbers",
]
