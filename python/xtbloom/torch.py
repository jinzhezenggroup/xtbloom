"""PyTorch autograd integration for xTBloom through the DLPack bridge.

:func:`xtbloom_torch` runs packed xTBloom inference on PyTorch tensors (host CPU
or CUDA device) with zero-copy DLPack input and output, and exposes exactly
one analytic gradient: ``dE/dR = -F`` with respect to the atomic positions,
which the native library already evaluates.

The autograd contract is intentionally narrow.

* Only ``positions`` may require gradient.  Requesting autograd for any other
  tensor input (atomic numbers, molecular charge, ``uhf``, spin channels, and
  the optional point-charge/response groups) raises
  :class:`XTBloomNotSupportedError` eagerly at forward time, because xTBloom
  does not compute those derivatives.
* Gradient flow through the ``forces`` output (the force Hessian ``dF/dR``)
  raises :class:`XTBloomNotSupportedError` during backward.
* Higher-order differentiation is rejected explicitly instead of returning a
  partial or zero Hessian, because the native force derivative is unavailable.
* Every tensor input is detached before the native DLPack call, so calling
  :func:`xtbloom_torch` never participates in or mutates an existing autograd
  graph beyond the one this op creates.

PyTorch is never imported by the rest of xTBloom, and this module imports it
only lazily inside :func:`xtbloom_torch` (through ``importlib``), preserving the
package's "no runtime torch import" guarantee.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from .exceptions import XTBloomNotSupportedError, XTBloomValueError
from .interface import compute_arrays

if TYPE_CHECKING:
    from types import ModuleType

_FUNCTION_CLASS: object | None = None


class _Tensor(Protocol):
    """Structural type for the torch.Tensor members the op actually uses.

    The protocol is type-checking only; xTBloom never imports torch at runtime
    unless :func:`xtbloom_torch` is called.  Only the small, dtype/device-safe
    surface below is declared.
    """

    dtype: object
    shape: tuple[int, ...]
    device: object
    is_cuda: bool

    def is_floating_point(self) -> bool: ...

    def dim(self) -> int: ...

    def detach(self) -> _Tensor: ...

    def clone(self) -> _Tensor: ...

    def contiguous(self) -> _Tensor: ...

    def to(self, *, device: object = ..., dtype: object = ...) -> _Tensor: ...

    def index_select(self, dim: int, index: _Tensor) -> _Tensor: ...

    def unsqueeze(self, dim: int) -> _Tensor: ...

    def __neg__(self) -> _Tensor: ...

    def __mul__(self, other: object) -> _Tensor: ...

    def __sub__(self, other: object) -> _Tensor: ...

    def __getitem__(self, index: object) -> _Tensor: ...


class _AutogradFunction(Protocol):
    """Structural type for the ``torch.autograd.Function.apply`` entry point."""

    def apply(self, *args: object) -> tuple[object, object]: ...


class _FunctionCtx(Protocol):
    """Structural type for the autograd context attributes saved by forward."""

    _result: object | None
    _forces: _Tensor
    _atom_offsets: _Tensor

    def set_materialize_grads(self, value: bool) -> None: ...


def _torch() -> ModuleType:
    """Import PyTorch on first use; the rest of xTBloom never imports it."""
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise XTBloomNotSupportedError(
            "xtbloom_torch requires PyTorch, which is an optional integration "
            "and not an xTBloom dependency"
        ) from exc


def _normalize_layout(value: object) -> object:
    """Return a detached, compact C-contiguous array of the same dtype.

    The DLPack bridge only ever exports borrowed views (torch's ``copy=True``
    does not actually pack strided views into a contiguous copy), so the op
    itself produces the contiguous copy for non-contiguous torch tensors and
    numpy arrays before the native descriptors are bound.  Contiguous inputs
    pass through without a copy; scalar types are never coerced.
    """
    torch = _torch()
    if torch.is_tensor(value):
        return cast("_Tensor", value).detach().contiguous()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    return value


def _to_tensor(value: object) -> _Tensor:
    """Import one finished result array as a PyTorch tensor without a copy."""
    torch = _torch()
    if torch.is_tensor(value):
        return cast("_Tensor", value)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value))
    return torch.from_dlpack(value)


def _function() -> _AutogradFunction:
    """Return the cached ``torch.autograd.Function`` subclass for this op.

    The subclass is defined after PyTorch is imported so the module keeps its
    lazy-import guarantee; only the first call pays for the definition.
    """
    global _FUNCTION_CLASS
    if _FUNCTION_CLASS is not None:
        return cast("_AutogradFunction", _FUNCTION_CLASS)

    torch = _torch()

    class _XTBloomTorchFunction(torch.autograd.Function):
        """One xTBloom forward/backward pair restricted to the dR gradient."""

        @staticmethod
        def forward(
            ctx: _FunctionCtx,
            positions: _Tensor,
            atomic_numbers: object,
            atom_offsets: object,
            molecular_charges: object,
            unpaired_electrons: object,
            spin_channels: object,
            backend: str | int,
            device_id: int | None,
            cpu_threads: int,
            stream: int | None,
            max_scc_iterations: int,
            charge_tolerance: float,
            energy_tolerance: float,
            electronic_temperature: float,
        ) -> tuple[_Tensor, _Tensor]:
            # The C ABI and the DLPack bridge are deliberately strict about
            # dtype/layout, so validate the couple of autograd-relevant facts
            # here and let the native path reject everything else.
            if not (torch.is_tensor(positions) and positions.is_floating_point()):
                raise XTBloomValueError(
                    "positions must be a floating-point PyTorch tensor"
                )
            if positions.dtype != torch.float64:
                raise XTBloomValueError(
                    f"xtbloom_torch requires float64 positions, got {positions.dtype}"
                )
            if positions.dim() != 2 or positions.shape[1] != 3:
                raise XTBloomValueError("positions must have shape (natoms, 3)")
            for name, value in (
                ("atomic_numbers", atomic_numbers),
                ("atom_offsets", atom_offsets),
                ("molecular_charges", molecular_charges),
                ("unpaired_electrons", unpaired_electrons),
                ("spin_channels", spin_channels),
            ):
                if torch.is_tensor(value) and bool(
                    getattr(value, "requires_grad", False)
                ):
                    raise XTBloomNotSupportedError(
                        "xtbloom_torch supports autograd only for positions; "
                        f"gradients w.r.t. {name} are not computed"
                    )

            # Normalize layout and detach everything before the native call so
            # the op never participates in an outside autograd graph and the
            # strict zero-copy DLPack descriptors are always satisfied.
            # Import offsets through DLPack once and pass the resulting Torch
            # tensor to both native inference and backward.  In particular,
            # CUDA producers such as CuPy intentionally reject np.asarray.
            normalized_atom_offsets = _to_tensor(_normalize_layout(atom_offsets))
            result = compute_arrays(
                atom_offsets=normalized_atom_offsets,
                atomic_numbers=_normalize_layout(atomic_numbers),
                positions=_normalize_layout(positions),
                molecular_charges=_normalize_layout(molecular_charges),
                unpaired_electrons=_normalize_layout(unpaired_electrons),
                spin_channels=_normalize_layout(spin_channels),
                copy=False,
                backend=backend,
                device_id=device_id,
                cpu_threads=cpu_threads,
                stream=stream,
                max_scc_iterations=max_scc_iterations,
                charge_tolerance=charge_tolerance,
                energy_tolerance=energy_tolerance,
                electronic_temperature=electronic_temperature,
                result_memory="cuda" if positions.is_cuda else "host",
            )
            energies = _to_tensor(result.energies)
            forces = _to_tensor(result.forces)
            # Keep host keepalive arrays / native result arenas alive for as
            # long as the returned tensors (and the saved clone below) alias
            # them.  Backward needs its own private snapshot of positions'
            # gradient source (-forces) so later in-place user edits of the
            # returned tensors cannot corrupt the gradient.
            ctx._result = result
            ctx._forces = forces.detach().clone()
            ctx._atom_offsets = normalized_atom_offsets.detach().clone()
            # An energy-only loss has no gradient for the forces output.  Keep
            # that state as None so CUDA backward avoids materializing and
            # scanning a full zero tensor solely to distinguish an unused
            # output from a real force-gradient request.
            ctx.set_materialize_grads(False)
            return energies, forces

        @staticmethod
        def backward(
            ctx: _FunctionCtx,
            grad_energies: _Tensor | None,
            grad_forces: _Tensor | None,
        ) -> tuple[_Tensor | None, ...]:
            # create_graph=True enables grad mode while custom backward runs.
            # Reject it immediately: ctx._forces is a detached native result,
            # so allowing this path would silently omit dF/dR and report a
            # partial or all-zero Hessian.
            if torch.is_grad_enabled():
                raise XTBloomNotSupportedError(
                    "xtbloom_torch does not support higher-order "
                    "differentiation because dF/dR is not computed"
                )
            if grad_forces is not None:
                raise XTBloomNotSupportedError(
                    "xtbloom_torch does not support gradients through the forces "
                    "output (the force Hessian dF/dR is not computed); only the "
                    "energy gradient dE/dR = -F is available"
                )
            if grad_energies is None:
                return (None,) * 14
            # dE/dR = -F, block-diagonal over the ragged batch: atom a of
            # system i receives -grad_energy[i] * F_a.
            offsets = ctx._atom_offsets.to(device=ctx._forces.device, dtype=torch.int64)
            counts = offsets[1:] - offsets[:-1]
            system_of_atom = torch.repeat_interleave(
                torch.arange(offsets.shape[0] - 1, device=offsets.device), counts
            )
            per_atom = (
                grad_energies.to(device=ctx._forces.device, dtype=ctx._forces.dtype)
                .index_select(0, system_of_atom)
                .unsqueeze(1)
            )
            grad_positions = -per_atom * ctx._forces
            return (
                grad_positions,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    _XTBloomTorchFunction.__name__ = "XTBloomTorchFunction"
    _FUNCTION_CLASS = _XTBloomTorchFunction
    return cast("_AutogradFunction", _FUNCTION_CLASS)


def xtbloom_torch(
    positions: object,
    atomic_numbers: object,
    atom_offsets: object,
    molecular_charges: object,
    unpaired_electrons: object,
    spin_channels: object | None = None,
    *,
    backend: str | int = "auto",
    device_id: int | None = None,
    cpu_threads: int = 1,
    stream: int | None = None,
    max_scc_iterations: int = 250,
    charge_tolerance: float = 1.0e-6,
    energy_tolerance: float = 1.0e-8,
    electronic_temperature: float = 300.0,
) -> tuple[object, object]:
    """Run xTBloom inference on PyTorch tensors with a ``dR``-only autograd op.

    The inputs mirror the packed DLPack descriptors of
    :func:`xtbloom.compute_arrays`; ``positions`` is the only differentiable
    tensor and the only argument that may set ``requires_grad=True``.  The
    returned ``(energies, forces)`` pair follows the same units as the rest of
    xTBloom (Hartree and Hartree/bohr).

    Parameters
    ----------
    positions : (natoms, 3) float64 torch.Tensor
        Cartesian coordinates in bohr.  The only input supporting autograd;
        its analytic gradient is ``dE/dR = -F``.
    atomic_numbers : (natoms,) int32
        Atomic numbers; a torch tensor or any DLPack producer (for example a
        numpy array).
    atom_offsets : (nsystems + 1,) int64
        Ragged atom offsets; ``offsets[-1]`` is the total atom count.
    molecular_charges : (nsystems,) float64
        Total molecular charge of each system.
    unpaired_electrons : (nsystems,) int32
        Number of unpaired electrons of each system.
    spin_channels : (nsystems,) int32, optional
        Orbital channels (1 restricted / 2 unrestricted); defaults to all
        restricted ``1``, exactly like :class:`xtbloom.ArrayBatch`.
    backend, device_id, cpu_threads, stream
        Same context selection as :class:`xtbloom.ArrayBatch`.
    max_scc_iterations, charge_tolerance, energy_tolerance, electronic_temperature
        Same SCC options as :class:`xtbloom.ArrayBatch`.  ``electronic_temperature``
        is given in kelvin.

    Returns
    -------
    energies : (nsystems,) float64 torch.Tensor
        Per-system energies in Hartree.
    forces : (natoms, 3) float64 torch.Tensor
        Per-atom forces in Hartree/bohr.

    Raises
    ------
    XTBloomNotSupportedError
        If PyTorch is unavailable, if a non-``positions`` tensor requests
        autograd, if gradient flows through the ``forces`` output, or if
        higher-order differentiation is requested.
    XTBloomValueError
        If ``positions`` is not a ``float64`` tensor of shape ``(natoms, 3)``.
    """
    _torch()
    function = _function()
    return function.apply(
        positions,
        atomic_numbers,
        atom_offsets,
        molecular_charges,
        unpaired_electrons,
        spin_channels,
        backend,
        device_id,
        int(cpu_threads),
        stream,
        int(max_scc_iterations),
        float(charge_tolerance),
        float(energy_tolerance),
        float(electronic_temperature),
    )
