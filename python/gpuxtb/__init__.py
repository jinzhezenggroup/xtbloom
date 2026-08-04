"""gpuxtb: batched GFN-xTB energy and analytic-force inference.

The Python package wraps the public gpuxtb C ABI through :mod:`ctypes`,
marshals numpy arrays between Python and the C descriptors, and exposes a
tblite-like high-level interface (:class:`gpuxtb.interface.Calculator`),
the native ragged-batch model (:class:`gpuxtb.interface.BatchCalculator`),
plus optional ASE (:mod:`gpuxtb.ase`) and dpdata
(:mod:`gpuxtb.plugins.dpdata`) integrations.

Atomic units everywhere: bohr, Hartree, Hartree/bohr, elementary charge.
"""

__version__ = "0.1.0"

from .exceptions import (
    GPUxtbError,
    GPUxtbNotSupportedError,
    GPUxtbRuntimeError,
    GPUxtbValueError,
)
from .interface import (
    BatchCalculator,
    BatchResult,
    Calculator,
    Context,
    PointCharge,
    Result,
    Structure,
    numbers_to_symbols,
    symbols_to_numbers,
)

__all__ = [
    "__version__",
    "Calculator",
    "BatchCalculator",
    "Structure",
    "Result",
    "BatchResult",
    "PointCharge",
    "Context",
    "symbols_to_numbers",
    "numbers_to_symbols",
    "GPUxtbError",
    "GPUxtbRuntimeError",
    "GPUxtbValueError",
    "GPUxtbNotSupportedError",
]