"""PyTorch autograd integration for xTBloom through a compiled stable-ABI op.

:func:`xtbloom_torch` runs packed xTBloom inference on PyTorch tensors (host CPU
or CUDA device) with zero copy, and exposes exactly one analytic gradient:
``dE/dR = -F`` with respect to the atomic positions, which the native library
already evaluates.

The native data plane lives in a compiled torch extension,
``libxtbloom_torch_ext`` (built from
``src/bindings/torch/xtbloom_torch_ext.cpp``), which is
written against the **LibTorch Stable ABI** (torch >= 2.10): it binds torch
tensor data pointers directly to the public xtbloom C ABI descriptors and runs
CPU and host-output calls synchronously. CUDA device outputs transparently
follow ``torch.cuda.current_stream()``, so callers receive the same ordinary
stream-ordered tensors as from other CUDA-enabled PyTorch operations. Native
scheduling and lifetime management remain implementation details. One binary
works across torch releases and is loaded lazily through
``torch.ops.load_library``; the Python module below only supplies the thin
autograd ``Function`` and the ``torch.compile`` graph-break shim.

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
* Every tensor input is detached before the native call, so calling
  :func:`xtbloom_torch` never participates in or mutates an existing autograd
  graph beyond the one this op creates.

PyTorch is never imported by the rest of xTBloom, and this module imports it
only lazily inside :func:`xtbloom_torch` (through ``importlib``), preserving the
package's "no runtime torch import" guarantee.  The compiled extension is a
plain shared library loaded by torch, so ``import xtbloom`` likewise does not
load either torch or the extension.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import operator
import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, SupportsIndex, cast

import numpy as np

from . import library
from .exceptions import XTBloomNotSupportedError, XTBloomRuntimeError, XTBloomValueError

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

_FUNCTION_CLASS: object | None = None
_CUDA_PROCESS_ID: int | None = None


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
    _version: int

    def is_floating_point(self) -> bool: ...

    def fill_(self, value: object) -> _Tensor: ...

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
    """Structural type for the autograd context state used by this op."""

    saved_tensors: tuple[_Tensor, ...]
    _xtbloom_submission_id: int

    def set_materialize_grads(self, value: bool) -> None: ...

    def save_for_backward(self, *tensors: _Tensor) -> None: ...


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

    This helper handles inputs whose concrete array type is already known.
    Arbitrary DLPack producers must first be imported as tensors by
    :func:`_to_compact_tensor`, because their layout cannot be inspected here.
    Contiguous inputs pass through without a copy; scalar types are never
    coerced.
    """
    torch = _torch()
    if torch.is_tensor(value):
        return cast("_Tensor", value).detach().contiguous()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    return value


def _to_tensor(value: object) -> _Tensor:
    """Import an auxiliary array or DLPack producer as a tensor without a copy."""
    torch = _torch()
    if torch.is_tensor(value):
        return cast("_Tensor", value)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value))
    return torch.from_dlpack(value)


def _to_compact_tensor(value: object) -> tuple[_Tensor, _Tensor]:
    """Import one auxiliary and return its compact view plus lifetime owner.

    A non-Torch DLPack producer may export a strided tensor, so layout
    normalization has to happen *after* ``torch.from_dlpack``.  Keep the
    imported tensor separately: CUDA ``contiguous()`` may enqueue a copy, and
    the native request pool must retain the borrowed DLPack allocation until
    that copy and the dependent inference have completed.
    """
    imported = _to_tensor(value).detach()
    return imported.contiguous(), imported


# --- compiled torch extension -------------------------------------------------
#
# The native data plane is libxtbloom_torch_ext (LibTorch Stable ABI), a plain
# shared library shipped next to libxtbloom and loaded once per process through
# torch.ops.load_library.  It is not a CPython module, so the wheel remains a
# single pure-``py3`` archive and no torch headers are needed beyond building
# the extension itself.

_TORCH_EXT_LOADED = False


def _torch_extension_path() -> Path | None:
    """Locate ``libxtbloom_torch_ext`` next to the resolved ``libxtbloom``.

    CMake installs the extension into the same ``lib`` directory as
    ``libxtbloom``.  Mirror ``library.library_path`` resolution so an explicit
    ``XTBLOOM_LIBRARY`` (or an installed wheel whose native libraries sit next
    to the Python package) still yields the extension when the Python package
    itself is imported from a source tree on ``PYTHONPATH``.
    """
    runtime_dirs: list[Path] = []
    with contextlib.suppress(library.XTBloomRuntimeError):
        runtime_dirs.append(Path(library.library_path()).resolve().parent)
    package_dir = Path(__file__).resolve().parent
    for runtime_dir in (
        package_dir / "lib",
        package_dir / "lib64",
        package_dir / "bin",
        package_dir,
    ):
        if runtime_dir not in runtime_dirs:
            runtime_dirs.append(runtime_dir)
    for runtime_dir in runtime_dirs:
        for pattern in (
            "libxtbloom_torch_ext*.so*",
            "libxtbloom_torch_ext*.dylib*",
            "xtbloom_torch_ext*.dll",
            "libxtbloom_torch_ext*.dll",
        ):
            matches = sorted(runtime_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


def _xtbloom_torch_op() -> Callable[..., tuple[object, object, int]]:
    """Return (and load on first use) the private compiled forward operator."""
    global _TORCH_EXT_LOADED
    torch = _torch()
    if not _TORCH_EXT_LOADED:
        # Establish the same native runtime selected by the rest of the Python
        # package and preload optional providers such as scipy-openblas32. The
        # compiled op intentionally resolves xtbloom itself, but must not bypass
        # Python's provider discovery when it is the first public API called.
        library.load_library()
        path = _torch_extension_path()
        if path is None:
            raise XTBloomNotSupportedError(
                "xtbloom_torch requires the compiled torch extension "
                "(libxtbloom_torch_ext), which is not installed next to this "
                "xtbloom package; rebuild/install xtbloom with CMake support for "
                "Torch >= 2.10 so the extension is bundled"
            )
        try:
            torch.ops.load_library(str(path))
        except OSError as exc:
            raise XTBloomRuntimeError(
                "xtbloom_torch could not load the torch extension: this build "
                "of libxtbloom_torch_ext may be incompatible with the installed "
                "torch"
            ) from exc
        _TORCH_EXT_LOADED = True
    return cast(
        "Callable[..., tuple[object, object, int]]",
        torch.ops.xtbloom._xtbloom_torch_forward,
    )


_METHOD_ALIASES = {
    "GFN1-xTB": library.MODEL_GFN1_XTB,
    "GFN1": library.MODEL_GFN1_XTB,
    "GFN2-xTB": library.MODEL_GFN2_XTB,
    "GFN2": library.MODEL_GFN2_XTB,
}

_BACKEND_ALIASES = {
    "auto": library.BACKEND_AUTO,
    "cpu": library.BACKEND_CPU,
    "cuda": library.BACKEND_CUDA,
}


def _resolve_method(method: str) -> int:
    """Validate the GFN method selector and return its public C-ABI model tag."""
    try:
        return _METHOD_ALIASES[method]
    except KeyError:
        raise XTBloomValueError(f"unknown method {method!r}") from None


def _resolve_backend(backend: str | int) -> int:
    """Validate the backend selector and return its int32 ABI value."""
    if isinstance(backend, str):
        try:
            return _BACKEND_ALIASES[backend]
        except KeyError:
            raise XTBloomValueError(f"unknown backend {backend!r}") from None
    value = int(backend)
    if value not in (
        library.BACKEND_AUTO,
        library.BACKEND_CPU,
        library.BACKEND_CUDA,
    ):
        raise XTBloomValueError(f"unknown backend {backend!r}")
    return value


def _resolve_context_scalars(
    device_id: int | None,
    cpu_threads: int,
) -> tuple[int, int]:
    """Resolve the context scalar arguments exactly like ``interface.Context``."""
    resolved_device = -1 if device_id is None else int(device_id)
    resolved_threads = int(cpu_threads)
    if resolved_threads < 0:
        raise XTBloomValueError("cpu_threads must be nonnegative")
    return resolved_device, resolved_threads


def _resolve_scc_policy(
    scc_mixer: str | int,
    scc_mixer_history: int,
    scc_mixer_damping: float,
    determinism: str | int,
) -> tuple[int, int, float, int]:
    """Validate the ABI-v3 SCC policy without importing the high-level interface."""
    mixer_aliases = {"modified_broyden": library.SCC_MIXER_MODIFIED_BROYDEN}
    determinism_aliases = {
        "default": library.DETERMINISM_DEFAULT,
        "reproducible": library.DETERMINISM_REPRODUCIBLE,
    }

    def exact_integer(name: str, value: object) -> int:
        if isinstance(value, bool | np.bool_):
            raise XTBloomValueError(f"{name} must be an integer")
        try:
            return int(operator.index(cast("SupportsIndex", value)))
        except TypeError:
            raise XTBloomValueError(f"{name} must be an integer") from None

    if isinstance(scc_mixer, str):
        try:
            mixer = mixer_aliases[scc_mixer]
        except KeyError:
            raise XTBloomValueError(
                "scc_mixer must be 'modified_broyden' or SCC_MIXER_MODIFIED_BROYDEN"
            ) from None
    else:
        mixer = exact_integer("scc_mixer", scc_mixer)
    if mixer != library.SCC_MIXER_MODIFIED_BROYDEN:
        raise XTBloomValueError(
            "scc_mixer must be 'modified_broyden' or SCC_MIXER_MODIFIED_BROYDEN"
        )

    history = exact_integer("scc_mixer_history", scc_mixer_history)
    if not 1 <= history <= library.MAX_SCC_MIXER_HISTORY:
        raise XTBloomValueError(
            f"scc_mixer_history must lie between 1 and {library.MAX_SCC_MIXER_HISTORY}"
        )

    damping = float(scc_mixer_damping)
    if not math.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise XTBloomValueError("scc_mixer_damping must be finite and lie in (0, 1]")

    if isinstance(determinism, str):
        try:
            deterministic = determinism_aliases[determinism]
        except KeyError:
            raise XTBloomValueError(
                "determinism must be 'default', 'reproducible', "
                "DETERMINISM_DEFAULT, or DETERMINISM_REPRODUCIBLE"
            ) from None
    else:
        deterministic = exact_integer("determinism", determinism)
    if deterministic not in (
        library.DETERMINISM_DEFAULT,
        library.DETERMINISM_REPRODUCIBLE,
    ):
        raise XTBloomValueError(
            "determinism must be 'default', 'reproducible', DETERMINISM_DEFAULT, "
            "or DETERMINISM_REPRODUCIBLE"
        )
    library.require_compute_options_v3(mixer, history, damping, deterministic)
    return mixer, history, damping, deterministic


def _check_cuda_process() -> None:
    """Reject inherited CUDA/pool state before touching any input producer."""
    if _CUDA_PROCESS_ID is not None and os.getpid() != _CUDA_PROCESS_ID:
        raise XTBloomNotSupportedError(
            "xtbloom_torch does not support use after CUDA state was inherited by "
            "fork; create workers before the first CUDA call or use a spawn-based "
            "multiprocessing start method"
        )


def _current_cuda_stream(
    torch: ModuleType,
    tensors: tuple[_Tensor, ...],
    *,
    backend: int,
    device_id: int,
) -> int:
    """Return the active Torch CUDA stream for the tensors' shared device.

    Importing a foreign DLPack producer into Torch establishes readiness on
    Torch's current stream. The compiled op then binds raw tensor pointers, so
    native work must remain on that stream. The raw handle is an implementation
    detail and is never accepted from the Python caller.
    """
    global _CUDA_PROCESS_ID
    _check_cuda_process()
    cuda_tensors = [tensor for tensor in tensors if bool(tensor.is_cuda)]
    if backend == library.BACKEND_CPU:
        return 0
    process_id = os.getpid()
    if cuda_tensors:
        device = cuda_tensors[0].device
        if any(tensor.device != device for tensor in cuda_tensors[1:]):
            raise XTBloomValueError("all CUDA tensors must be on the same device")
    else:
        # CUDA and AUTO also accept all-host descriptors. There is no tensor
        # device to infer from in that case, so pass the selected/current Torch
        # stream as a candidate. The native op discards it only when AUTO
        # resolves to the documented CPU fallback.
        if not bool(torch.cuda.is_available()):
            return 0
        device = device_id if device_id >= 0 else torch.cuda.current_device()

    current_stream = torch.cuda.current_stream(device)
    _CUDA_PROCESS_ID = process_id
    current_handle = int(current_stream.cuda_stream)
    return current_handle if current_handle > 0 else 0


def _native_forward(
    *,
    positions: object,
    atomic_numbers: object,
    atom_offsets: object,
    molecular_charges: object,
    unpaired_electrons: object,
    spin_channels: object,
    atomic_numbers_owner: object,
    atom_offsets_owner: object,
    molecular_charges_owner: object,
    unpaired_electrons_owner: object,
    spin_channels_owner: object,
    atomic_numbers_version: int,
    atom_offsets_version: int,
    molecular_charges_version: int,
    unpaired_electrons_version: int,
    spin_channels_version: int,
    out_energies: object,
    out_forces: object,
    model: int,
    backend: int,
    device_id: int,
    cpu_threads: int,
    stream: int,
    max_scc_iterations: int,
    charge_tolerance: float,
    energy_tolerance: float,
    electronic_temperature: float,
    scc_mixer: int,
    scc_mixer_history: int,
    scc_mixer_damping: float,
    determinism: int,
) -> tuple[object, object, int]:
    """Run the compiled stable-ABI op on Torch's selected execution stream.

    This is the only native call site of the module, so tests can substitute it
    to inject failures. ``electronic_temperature`` is in kelvin; the op
    converts it to the native k_B*T scale.
    """
    return _xtbloom_torch_op()(
        positions,
        atomic_numbers,
        atom_offsets,
        molecular_charges,
        unpaired_electrons,
        spin_channels,
        atomic_numbers_owner,
        atom_offsets_owner,
        molecular_charges_owner,
        unpaired_electrons_owner,
        spin_channels_owner,
        atomic_numbers_version,
        atom_offsets_version,
        molecular_charges_version,
        unpaired_electrons_version,
        spin_channels_version,
        out_energies,
        out_forces,
        int(model),
        backend,
        device_id,
        cpu_threads,
        stream,
        int(max_scc_iterations),
        float(charge_tolerance),
        float(energy_tolerance),
        float(electronic_temperature),
        int(scc_mixer),
        int(scc_mixer_history),
        float(scc_mixer_damping),
        int(determinism),
    )


def _native_wait(submission_id: int) -> None:
    """Settle one private CUDA submission before autograd consumes its result."""
    if submission_id == 0:
        return
    torch = _torch()
    cast("Callable[[int], None]", torch.ops.xtbloom._xtbloom_torch_wait)(
        int(submission_id)
    )


def _allocate_outputs(
    torch: ModuleType, device: object, nsystems: int, natoms: int
) -> tuple[_Tensor, _Tensor]:
    """Allocate failure-safe caller-returned tensors on ``device``.

    CUDA execution may finish after the Python call returns. Initializing the
    outputs to NaN ensures that a deferred call-level failure cannot expose
    uninitialized allocator bytes or values left by an earlier allocation.
    Successful native publication overwrites every requested element.
    """
    return (
        torch.full((nsystems,), float("nan"), dtype=torch.float64, device=device),
        torch.full((natoms, 3), float("nan"), dtype=torch.float64, device=device),
    )


def _preflight_positions(torch: ModuleType, positions: object) -> None:
    """Validate the autograd-relevant contract of the positions tensor."""
    positions_t = cast("_Tensor", positions)
    if not (torch.is_tensor(positions_t) and positions_t.is_floating_point()):
        raise XTBloomValueError("positions must be a floating-point PyTorch tensor")
    if positions_t.dtype != torch.float64:
        raise XTBloomValueError(
            f"xtbloom_torch requires float64 positions, got {positions_t.dtype}"
        )
    if positions_t.dim() != 2 or positions_t.shape[1] != 3:
        raise XTBloomValueError("positions must have shape (natoms, 3)")


def _reject_nonposition_grads(torch: ModuleType, values: dict[str, object]) -> None:
    """Reject autograd on any non-positions input eagerly."""
    for name, value in values.items():
        value_t = cast("_Tensor", value)
        if torch.is_tensor(value_t) and bool(getattr(value, "requires_grad", False)):
            raise XTBloomNotSupportedError(
                "xtbloom_torch supports autograd only for positions; "
                f"gradients w.r.t. {name} are not computed"
            )


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
            model: int,
            backend: str | int,
            device_id: int | None,
            cpu_threads: int,
            max_scc_iterations: int,
            charge_tolerance: float,
            energy_tolerance: float,
            electronic_temperature: float,
            scc_mixer: str | int,
            scc_mixer_history: int,
            scc_mixer_damping: float,
            determinism: str | int,
        ) -> tuple[_Tensor, _Tensor]:
            # The C ABI and the DLPack bridge are deliberately strict about
            # dtype/layout, so validate the couple of autograd-relevant facts
            # here and let the native path reject everything else.
            _check_cuda_process()
            _preflight_positions(torch, positions)
            _reject_nonposition_grads(
                torch,
                {
                    "atomic_numbers": atomic_numbers,
                    "atom_offsets": atom_offsets,
                    "molecular_charges": molecular_charges,
                    "unpaired_electrons": unpaired_electrons,
                    "spin_channels": spin_channels,
                },
            )

            # Normalize layout and detach everything before the native call so
            # the op never participates in an outside autograd graph and the
            # strict zero-copy tensor contract is always satisfied.  Import
            # offsets through DLPack once and pass the resulting Torch tensor
            # to both native inference and backward.  In particular, CUDA
            # producers such as CuPy intentionally reject np.asarray.
            normalized_atomic_numbers, atomic_numbers_owner = _to_compact_tensor(
                atomic_numbers
            )
            normalized_atom_offsets, atom_offsets_owner = _to_compact_tensor(
                atom_offsets
            )
            normalized_positions = cast("_Tensor", _normalize_layout(positions))
            normalized_molecular_charges, molecular_charges_owner = _to_compact_tensor(
                molecular_charges
            )
            normalized_unpaired_electrons, unpaired_electrons_owner = (
                _to_compact_tensor(unpaired_electrons)
            )
            nsystems = int(normalized_atom_offsets.shape[0]) - 1
            if spin_channels is None:
                spin_channels = torch.ones(
                    nsystems, dtype=torch.int32, device=positions.device
                )
            normalized_spin_channels, spin_channels_owner = _to_compact_tensor(
                spin_channels
            )
            resolved_backend = _resolve_backend(backend)
            resolved_device, resolved_threads = _resolve_context_scalars(
                device_id, cpu_threads
            )
            mixer, history, damping, deterministic = _resolve_scc_policy(
                scc_mixer, scc_mixer_history, scc_mixer_damping, determinism
            )
            out_energies, out_forces = _allocate_outputs(
                torch,
                positions.device,
                nsystems,
                int(positions.shape[0]),
            )
            resolved_stream = _current_cuda_stream(
                torch,
                (
                    normalized_positions,
                    normalized_atomic_numbers,
                    normalized_atom_offsets,
                    normalized_molecular_charges,
                    normalized_unpaired_electrons,
                    normalized_spin_channels,
                    out_energies,
                    out_forces,
                ),
                backend=resolved_backend,
                device_id=resolved_device,
            )
            _native_result = _native_forward(
                positions=normalized_positions,
                atomic_numbers=normalized_atomic_numbers,
                atom_offsets=normalized_atom_offsets,
                molecular_charges=normalized_molecular_charges,
                unpaired_electrons=normalized_unpaired_electrons,
                spin_channels=normalized_spin_channels,
                atomic_numbers_owner=atomic_numbers_owner,
                atom_offsets_owner=atom_offsets_owner,
                molecular_charges_owner=molecular_charges_owner,
                unpaired_electrons_owner=unpaired_electrons_owner,
                spin_channels_owner=spin_channels_owner,
                atomic_numbers_version=int(normalized_atomic_numbers._version),
                atom_offsets_version=int(normalized_atom_offsets._version),
                molecular_charges_version=int(normalized_molecular_charges._version),
                unpaired_electrons_version=int(normalized_unpaired_electrons._version),
                spin_channels_version=int(normalized_spin_channels._version),
                out_energies=out_energies,
                out_forces=out_forces,
                model=model,
                backend=resolved_backend,
                device_id=resolved_device,
                cpu_threads=resolved_threads,
                stream=resolved_stream,
                max_scc_iterations=max_scc_iterations,
                charge_tolerance=charge_tolerance,
                energy_tolerance=energy_tolerance,
                electronic_temperature=electronic_temperature,
                scc_mixer=mixer,
                scc_mixer_history=history,
                scc_mixer_damping=damping,
                determinism=deterministic,
            )
            energies, forces, submission_id = cast(
                "tuple[_Tensor, _Tensor, int]", _native_result
            )
            ctx._xtbloom_submission_id = int(submission_id)
            # Backward needs its own private snapshot of positions' gradient
            # source (-forces) so later in-place user edits of the returned
            # tensors cannot corrupt the gradient.
            ctx.save_for_backward(
                forces.detach().clone(), normalized_atom_offsets.detach().clone()
            )
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
            # Reject it immediately: the saved forces are a detached native
            # result, so allowing this path would silently omit dF/dR and
            # report a partial or all-zero Hessian.
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
                return (None,) * 18
            _native_wait(ctx._xtbloom_submission_id)
            saved_forces, saved_atom_offsets = ctx.saved_tensors
            # dE/dR = -F, block-diagonal over the ragged batch: atom a of
            # system i receives -grad_energy[i] * F_a.
            offsets = saved_atom_offsets.to(
                device=saved_forces.device, dtype=torch.int64
            )
            counts = offsets[1:] - offsets[:-1]
            system_of_atom = torch.repeat_interleave(
                torch.arange(offsets.shape[0] - 1, device=offsets.device), counts
            )
            per_atom = (
                grad_energies.to(device=saved_forces.device, dtype=saved_forces.dtype)
                .index_select(0, system_of_atom)
                .unsqueeze(1)
            )
            grad_positions = -per_atom * saved_forces
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
                None,
                None,
                None,
                None,
            )

    _XTBloomTorchFunction.__name__ = "XTBloomTorchFunction"
    _FUNCTION_CLASS = _XTBloomTorchFunction
    return cast("_AutogradFunction", _FUNCTION_CLASS)


def _xtbloom_torch_impl(
    positions: object,
    atomic_numbers: object,
    atom_offsets: object,
    molecular_charges: object,
    unpaired_electrons: object,
    spin_channels: object | None = None,
    *,
    method: str = "GFN2-xTB",
    backend: str | int = "auto",
    device_id: int | None = None,
    cpu_threads: int = 1,
    max_scc_iterations: int = 250,
    charge_tolerance: float = 1.0e-6,
    energy_tolerance: float = 1.0e-8,
    electronic_temperature: float = 300.0,
    scc_mixer: str | int = "modified_broyden",
    scc_mixer_history: int = library.DEFAULT_SCC_MIXER_HISTORY,
    scc_mixer_damping: float = library.DEFAULT_SCC_MIXER_DAMPING,
    determinism: str | int = "default",
) -> tuple[object, object]:
    """Execute one packed xtbloom inference (the traceable-unsafe core).

    Kept private so it can be exposed through a ``torch.compile``-safe wrapper;
    Dynamo must never trace the compiled-op dispatch below.
    """
    _torch()
    model = _resolve_method(method)
    return _function().apply(
        positions,
        atomic_numbers,
        atom_offsets,
        molecular_charges,
        unpaired_electrons,
        spin_channels,
        model,
        backend,
        device_id,
        int(cpu_threads),
        int(max_scc_iterations),
        float(charge_tolerance),
        float(energy_tolerance),
        float(electronic_temperature),
        scc_mixer,
        scc_mixer_history,
        scc_mixer_damping,
        determinism,
    )


_XTBLOOM_TORCH_IMPL: Callable[..., tuple[object, object]] | None = None


def _disabled_torch_impl(torch: ModuleType) -> Callable[..., tuple[object, object]]:
    """Mark ``_xtbloom_torch_impl`` as opaque so Dynamo graph-breaks on it.

    Dynamo cannot trace xtbloom's native custom-op boundary, so the op is
    deliberately excluded from the compiler: the recursive form of
    ``torch._dynamo.disable`` uninstalls Dynamo's frame interception for the
    duration of the call and marks the callable opaque. Calling it inside
    ``torch.compile`` graph-breaks and executes it eagerly: correct results,
    no trace-time error, and no compilation speedup for the xtbloom call itself.
    """
    return cast(
        "Callable[..., tuple[object, object]]",
        torch._dynamo.disable(_xtbloom_torch_impl),
    )


def xtbloom_torch(
    positions: object,
    atomic_numbers: object,
    atom_offsets: object,
    molecular_charges: object,
    unpaired_electrons: object,
    spin_channels: object | None = None,
    *,
    method: str = "GFN2-xTB",
    backend: str | int = "auto",
    device_id: int | None = None,
    cpu_threads: int = 1,
    max_scc_iterations: int = 250,
    charge_tolerance: float = 1.0e-6,
    energy_tolerance: float = 1.0e-8,
    electronic_temperature: float = 300.0,
    scc_mixer: str | int = "modified_broyden",
    scc_mixer_history: int = library.DEFAULT_SCC_MIXER_HISTORY,
    scc_mixer_damping: float = library.DEFAULT_SCC_MIXER_DAMPING,
    determinism: str | int = "default",
) -> tuple[object, object]:
    """Run xTBloom inference on PyTorch tensors with a ``dR``-only autograd op.

    ``method`` accepts ``"GFN1-xTB"``/``"GFN1"`` and
    ``"GFN2-xTB"``/``"GFN2"``. GFN2-xTB remains the default for backward
    compatibility; both models use the same compiled stable-ABI data plane and
    the same ``dE/dR = -F`` autograd contract.

    The inputs mirror the packed ragged-batch descriptors of
    :class:`xtbloom.ArrayBatch`; ``positions`` is the only differentiable
    tensor and the only argument that may set ``requires_grad=True``.  The
    returned ``(energies, forces)`` pair follows the same units as the rest of
    xTBloom (Hartree and Hartree/bohr). CUDA execution follows
    ``torch.cuda.current_stream()``; the native stream handle is deliberately
    not part of the Python API.

    ``xtbloom_torch`` is eager-only by design: it dispatches a compiled custom
    operator that is kept opaque to Dynamo. Calling it inside ``torch.compile``
    therefore inserts a graph break and executes the call eagerly (correct
    results, no compilation speedup for the xtbloom call itself), instead of
    failing at trace time. CUDA work remains stream-ordered after that eager
    dispatch and does not force a host synchronization.

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
    method : str
        GFN model name: ``"GFN1-xTB"``/``"GFN1"`` or
        ``"GFN2-xTB"``/``"GFN2"``. Defaults to ``"GFN2-xTB"``.
    backend, device_id, cpu_threads
        Same context selection as :class:`xtbloom.ArrayBatch`.
    max_scc_iterations, charge_tolerance, energy_tolerance, electronic_temperature
        Same SCC options as :class:`xtbloom.ArrayBatch`.  ``electronic_temperature``
        is given in kelvin.
    scc_mixer, scc_mixer_history, scc_mixer_damping
        Low-level modified-Broyden policy. The default remains history 8 and
        damping 0.4; history is limited to 1..64 because CUDA workspace grows
        with both the history length and its square.
    determinism
        ``"default"`` or ``"reproducible"``. Reproducible mode requests stable
        replay only within the same backend, provider, toolkit, hardware, and
        process; it is not a cross-platform bitwise guarantee.

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
        If ``method`` is unknown or ``positions`` is not a ``float64`` tensor
        of shape ``(natoms, 3)``.
    """
    global _XTBLOOM_TORCH_IMPL
    torch = _torch()
    if _XTBLOOM_TORCH_IMPL is None:
        _XTBLOOM_TORCH_IMPL = _disabled_torch_impl(torch)
    return _XTBLOOM_TORCH_IMPL(
        positions,
        atomic_numbers,
        atom_offsets,
        molecular_charges,
        unpaired_electrons,
        spin_channels,
        method=method,
        backend=backend,
        device_id=device_id,
        cpu_threads=cpu_threads,
        max_scc_iterations=max_scc_iterations,
        charge_tolerance=charge_tolerance,
        energy_tolerance=energy_tolerance,
        electronic_temperature=electronic_temperature,
        scc_mixer=scc_mixer,
        scc_mixer_history=scc_mixer_history,
        scc_mixer_damping=scc_mixer_damping,
        determinism=determinism,
    )
