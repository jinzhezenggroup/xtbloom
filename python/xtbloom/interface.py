"""High-level Python interface to the xTBloom public C ABI.

The class layout intentionally mirrors ``tblite.interface`` (``Structure``,
``Calculator``, ``Result``) so that the package feels familiar to users of
``tblite``/``xtb-python``, while also exposing the native ragged-batch model of
the C API through :class:`BatchStructure`, :class:`BatchCalculator`, and
:class:`BatchResult`.

Atomic units: positions are in bohr, energies in Hartree, forces in
Hartree/bohr, and charges in elementary-charge units. The electronic
temperature is given in kelvin and converted internally.

Charge and spin semantics are implemented directly on top of the C ABI:

* ``charge`` maps to ``molecular_charges``.
* ``uhf`` (number of unpaired electrons, ``multiplicity - 1``) maps to
  ``unpaired_electrons``.
* ``spin_channels`` selects restricted (1) or unrestricted (2) orbitals. This
  high-level interface defaults open-shell systems to unrestricted and submits
  that explicit choice to either CPU or CUDA.
"""

from __future__ import annotations

import contextlib
import ctypes
import math
import operator
import struct
import typing
import weakref
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    SupportsFloat,
    SupportsIndex,
)

import numpy as np
import numpy.typing as npt

from . import _array as _array_adapter
from . import _dlpack, library
from .exceptions import XTBloomNotSupportedError, XTBloomRuntimeError, XTBloomValueError

if TYPE_CHECKING:
    import builtins
    from collections.abc import Callable, Sequence
    from types import TracebackType

# --- periodic-table helpers (mirrors tblite.interface) ---------------------------


ELEMENT_SYMBOLS = [
    *["H", "He"],
    *["Li", "Be", "B", "C", "N", "O", "F", "Ne"],
    *["Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"],
    *["K", "Ca"],
    *["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"],
    *["Ga", "Ge", "As", "Se", "Br", "Kr"],
    *["Rb", "Sr"],
    *["Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"],
    *["In", "Sn", "Sb", "Te", "I", "Xe"],
    *["Cs", "Ba"],
    *[
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
    ],
    *["Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"],
    *["Tl", "Pb", "Bi", "Po", "At", "Rn"],
    *["Fr", "Ra"],
    *[
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
    ],
    *["Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"],
    *["Nh", "Fl", "Mc", "Lv", "Ts", "Og"],
]

SYMBOL_TO_NUMBER = {symbol: number + 1 for number, symbol in enumerate(ELEMENT_SYMBOLS)}


def _as_integer(name: str, value: object) -> int:
    """Return an exact integer without silently truncating floats."""
    if isinstance(value, bool | np.bool_):
        raise XTBloomValueError(f"{name} must be an integer")
    try:
        # The runtime contract is validated below: only objects with a truthy
        # ``__index__`` succeed, and anything else raises XTBloomValueError.
        return int(operator.index(typing.cast("SupportsIndex", value)))
    except TypeError:
        raise XTBloomValueError(f"{name} must be an integer") from None


def symbols_to_numbers(symbols: Sequence[str]) -> list[int]:
    """Convert a list of atomic symbols to atomic numbers."""
    return [SYMBOL_TO_NUMBER[symbol] for symbol in symbols]


def numbers_to_symbols(numbers: Sequence[int]) -> list[str]:
    """Convert a list of atomic numbers to atomic symbols."""
    symbols = []
    for value in numbers:
        number = _as_integer("atomic number", value)
        if number < 1 or number > len(ELEMENT_SYMBOLS):
            raise XTBloomValueError("atomic numbers must lie between 1 and 118")
        symbols.append(ELEMENT_SYMBOLS[number - 1])
    return symbols


# --- supported methods -------------------------------------------------------------

_SUPPORTED_METHODS = {
    "GFN2-xTB": library.MODEL_GFN2_XTB,
    "GFN2": library.MODEL_GFN2_XTB,
}

_BACKENDS = {
    "auto": library.BACKEND_AUTO,
    "cpu": library.BACKEND_CPU,
    "cuda": library.BACKEND_CUDA,
}


def _default_spin_channels(uhf: int) -> int:
    """Pick the tblite-compatible spin-polarization default for a spin state."""
    return 2 if uhf != 0 else 1


def _resolve_uhf(uhf: int | None, multiplicity: int | None) -> int:
    """Resolve the number of unpaired electrons from ``uhf`` and/or ``multiplicity``."""
    if multiplicity is not None:
        multiplicity = _as_integer("multiplicity", multiplicity)
        if multiplicity < 1:
            raise XTBloomValueError("multiplicity must be a positive integer")
        unpaired = multiplicity - 1
        if uhf is not None and _as_integer("uhf", uhf) != unpaired:
            raise XTBloomValueError(
                f"uhf={uhf} is inconsistent with multiplicity={multiplicity}"
            )
        return unpaired
    unpaired = _as_integer("uhf", uhf) if uhf is not None else 0
    if unpaired < 0:
        raise XTBloomValueError("uhf must be nonnegative")
    return unpaired


# --- point charges ------------------------------------------------------------------


@dataclass(frozen=True)
class PointCharge:
    """An external point charge participating in every SCC iteration.

    Parameters
    ----------
    positions : (n, 3) array, bohr
        Cartesian positions of the point charges.
    charges : (n,) array, elementary-charge units
        Point-charge values.
    gammas : (n,) array, Hartree
        Explicit point-site screening (softened Coulomb parameter). This is a
        model parameter, not an optimizable degree of freedom; the C ABI
        requires it whenever point charges are present.
    """

    positions: np.ndarray
    charges: np.ndarray
    gammas: np.ndarray

    def __post_init__(self) -> None:
        """Validate and normalize point-charge arrays."""
        positions = np.asarray(self.positions, dtype=float)
        charges = np.asarray(self.charges, dtype=float)
        gammas = np.asarray(self.gammas, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise XTBloomValueError("point charge positions must have shape (n, 3)")
        if charges.shape != (len(positions),):
            raise XTBloomValueError("point charge values must match the position count")
        if gammas.shape != (len(positions),):
            raise XTBloomValueError("point charge gammas must match the position count")
        if len(positions) == 0:
            raise XTBloomValueError("point charges must be nonempty")
        if not (
            np.isfinite(positions).all()
            and np.isfinite(charges).all()
            and np.isfinite(gammas).all()
        ):
            raise XTBloomValueError("point charge inputs must be finite")
        object.__setattr__(self, "positions", np.ascontiguousarray(positions))
        object.__setattr__(self, "charges", np.ascontiguousarray(charges))
        object.__setattr__(self, "gammas", np.ascontiguousarray(gammas))


@dataclass(frozen=True)
class ChargeResponse:
    """A periodic per-atom SCC shift ``b`` and symmetric response matrix ``A``.

    For each molecule the periodic QM/MM coupling contributes an SCC shift
    ``b + A q`` on the atomic-charge channel and a variational energy
    ``q^T b + 0.5 q^T A q``, where ``q`` is the atomic charge vector. ``b`` is
    a per-atom potential shift (length ``n``) and ``A`` is the row-major
    symmetric response matrix with shape ``(n, n)``.  Both are treated as
    constant operators: derivatives of ``b`` and ``A`` with respect to
    coordinates are outside xTBloom and are not included in the forces.

    Parameters
    ----------
    shifts : (n,) array, Hartree/e
        Per-atom SCC potential shift ``b``.
    matrix : (n, n) array, Hartree/e^2
        Symmetric charge-response matrix ``A`` (validated for exact symmetry
        by the native compute call).
    """

    shifts: np.ndarray
    matrix: np.ndarray

    def __post_init__(self) -> None:
        """Validate and normalize the response operator arrays."""
        shifts = np.asarray(self.shifts, dtype=float)
        matrix = np.asarray(self.matrix, dtype=float)
        if shifts.ndim != 1:
            raise XTBloomValueError("charge response shifts must be one-dimensional")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise XTBloomValueError("charge response matrix must be square")
        if matrix.shape[0] != shifts.size:
            raise XTBloomValueError("charge response matrix must match the shift count")
        if not (np.isfinite(shifts).all() and np.isfinite(matrix).all()):
            raise XTBloomValueError("charge response inputs must be finite")
        object.__setattr__(self, "shifts", np.ascontiguousarray(shifts))
        object.__setattr__(self, "matrix", np.ascontiguousarray(matrix))


# --- structures --------------------------------------------------------------------


def _normalize_efield(efield: np.ndarray | list[float] | None) -> np.ndarray | None:
    """Validate and freeze a uniform external electric field vector.

    The field is a length-three vector in Hartree per elementary charge per
    bohr (atomic units). ``None`` leaves the structure field-free.
    """
    if efield is None:
        return None
    value = np.asarray(efield, dtype=float)
    if value.ndim != 1 or value.shape[0] != 3:
        raise XTBloomValueError("efield must be a vector of length 3")
    if not np.isfinite(value).all():
        raise XTBloomValueError("efield must be finite")
    if np.any(value != 0.0):
        return np.ascontiguousarray(value)
    return None


class Structure:
    """A molecular structure with an immutable atom count and atomic species.

    Coordinates (bohr), total charge, number of unpaired electrons, and spin
    channels can be updated through :meth:`update` without rebuilding the
    object.
    """

    def __init__(
        self,
        numbers: np.ndarray | list[int] | Sequence[str],
        positions: np.ndarray,
        charge: float = 0.0,
        uhf: int | None = None,
        multiplicity: int | None = None,
        spin_channels: int | None = None,
        point_charges: PointCharge | None = None,
        charge_response: ChargeResponse | None = None,
        efield: np.ndarray | list[float] | None = None,
    ) -> None:
        object_numbers = np.asarray(numbers, dtype=object)
        raw_numbers = np.asarray(numbers)
        if raw_numbers.ndim != 1 or raw_numbers.size == 0:
            raise XTBloomValueError("expected a nonempty one-dimensional numbers array")

        object_symbols = raw_numbers.dtype.kind == "O" and all(
            isinstance(value, str) for value in raw_numbers
        )
        if raw_numbers.dtype.kind in "US" or object_symbols:
            numbers = np.asarray(
                symbols_to_numbers([str(value) for value in raw_numbers]),
                dtype=np.int64,
            )
        else:
            if raw_numbers.dtype.kind in "bc" or (
                any(isinstance(value, bool | np.bool_) for value in object_numbers.flat)
            ):
                raise XTBloomValueError("atomic numbers must be exact integers")
            try:
                numeric_numbers = np.asarray(raw_numbers, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                raise XTBloomValueError(
                    "atomic numbers must be exact integers"
                ) from None
            if (
                not np.isfinite(numeric_numbers).all()
                or not np.equal(numeric_numbers, np.floor(numeric_numbers)).all()
            ):
                raise XTBloomValueError("atomic numbers must be exact integers")
            if (numeric_numbers < 1).any() or (numeric_numbers > 118).any():
                raise XTBloomValueError("atomic numbers must lie between 1 and 118")
            numbers = np.asarray(numeric_numbers, dtype=np.int64)
        positions = np.asarray(positions, dtype=float)

        if (numbers < 1).any() or (numbers > 118).any():
            raise XTBloomValueError("atomic numbers must lie between 1 and 118")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise XTBloomValueError("positions must have shape (n, 3)")
        if positions.shape[0] != numbers.size:
            raise XTBloomValueError("dimension mismatch between numbers and positions")
        if not np.isfinite(positions).all():
            raise XTBloomValueError("positions must be finite")

        unpaired = _resolve_uhf(uhf, multiplicity)
        if not math.isfinite(float(charge)):
            raise XTBloomValueError("charge must be finite")
        resolved_spin = (
            _default_spin_channels(unpaired)
            if spin_channels is None
            else _as_integer("spin_channels", spin_channels)
        )
        if resolved_spin not in (1, 2):
            raise XTBloomValueError(
                "spin_channels must be 1 (restricted) or 2 (unrestricted)"
            )

        self._numbers = np.ascontiguousarray(numbers, dtype=np.int32)
        self._positions = np.ascontiguousarray(positions, dtype=np.float64)
        self._charge = float(charge)
        self._uhf = unpaired
        self._spin_channels = resolved_spin
        self._point_charges = point_charges
        self._charge_response = charge_response
        self._efield = _normalize_efield(efield)

    def __len__(self) -> int:
        """Return the fixed atom count."""
        return int(self._numbers.size)

    @property
    def numbers(self) -> np.ndarray:
        """Atomic numbers (int32, length ``n``)."""
        return self._numbers

    @property
    def positions(self) -> np.ndarray:
        """Cartesian coordinates in bohr (float64, shape ``(n, 3)``)."""
        return self._positions

    @property
    def charge(self) -> float:
        """Total molecular charge in elementary-charge units."""
        return self._charge

    @property
    def uhf(self) -> int:
        """Number of unpaired electrons (``multiplicity - 1``)."""
        return self._uhf

    @property
    def multiplicity(self) -> int:
        """Total spin multiplicity."""
        return self._uhf + 1

    @property
    def spin_channels(self) -> int:
        """Orbital channels: 1 restricted or 2 unrestricted."""
        return self._spin_channels

    @property
    def point_charges(self) -> PointCharge | None:
        """External point charges attached to this structure, if any."""
        return self._point_charges

    @property
    def charge_response(self) -> ChargeResponse | None:
        """Periodic SCC shift ``b`` and response matrix ``A``, if any."""
        return self._charge_response

    @property
    def efield(self) -> np.ndarray | None:
        """Uniform external electric field in atomic units, if any.

        The field vector is a length-three float64 array (Hartree per
        elementary charge per bohr). ``None`` means no field attachment.
        """
        return self._efield

    def update(
        self,
        positions: np.ndarray | None = None,
        charge: float | None = None,
        uhf: int | None = None,
        multiplicity: int | None = None,
        spin_channels: int | None = None,
    ) -> None:
        """Update coordinates and electronic/spin state in place.

        Parameters
        ----------
        positions : optional, (n, 3) bohr
            Cartesian coordinates to replace the current ones.
        charge : optional, float
            Total molecular charge.
        uhf : optional, int
            Number of unpaired electrons.
        multiplicity : optional, int
            Total spin multiplicity (alternative to ``uhf``).
        spin_channels : optional, int
            Orbital channels (1 restricted / 2 unrestricted).
        """
        # Validate every candidate before mutating the object.  This keeps a
        # failed multi-field update transactional instead of leaving, for
        # example, new coordinates paired with an invalid spin request.
        next_positions = self._positions
        next_charge = self._charge
        next_uhf = self._uhf
        next_spin_channels = self._spin_channels

        if positions is not None:
            candidate_positions = np.asarray(positions, dtype=float)
            if candidate_positions.shape != self._positions.shape:
                raise XTBloomValueError(
                    "updated positions must keep the original shape"
                )
            if not np.isfinite(candidate_positions).all():
                raise XTBloomValueError("positions must be finite")
            next_positions = np.ascontiguousarray(candidate_positions, dtype=np.float64)
        if charge is not None:
            if not math.isfinite(float(charge)):
                raise XTBloomValueError("charge must be finite")
            next_charge = float(charge)
        if uhf is not None or multiplicity is not None:
            next_uhf = _resolve_uhf(uhf, multiplicity)
            if spin_channels is None:
                next_spin_channels = _default_spin_channels(next_uhf)
        if spin_channels is not None:
            next_spin_channels = _as_integer("spin_channels", spin_channels)
            if next_spin_channels not in (1, 2):
                raise XTBloomValueError(
                    "spin_channels must be 1 (restricted) or 2 (unrestricted)"
                )

        self._positions = next_positions
        self._charge = next_charge
        self._uhf = next_uhf
        self._spin_channels = next_spin_channels


# --- contexts ----------------------------------------------------------------------


def _destroy_native_context(
    library_instance: ctypes.CDLL, handle: ctypes.c_void_p
) -> None:
    """Destroy one native context; used by explicit close and finalization."""
    library_instance.xtbloom_context_destroy(handle)


class Context:
    """Ownership wrapper around a ``xtbloom_context_t``.

    The C context keeps its worker pools and numerical caches for the lifetime
    of the object, so a single context should be reused across geometry
    updates and steady-state calls. Creates and destroys the native context on
    enter/exit.
    """

    def __init__(
        self,
        backend: str | int = "auto",
        device_id: int | None = None,
        cpu_threads: int = 1,
        stream: int | None = None,
    ) -> None:
        if isinstance(backend, str):
            try:
                self._requested = _BACKENDS[backend]
            except KeyError:
                raise XTBloomValueError(f"unknown backend {backend!r}") from None
        else:
            if int(backend) not in (
                library.BACKEND_AUTO,
                library.BACKEND_CPU,
                library.BACKEND_CUDA,
            ):
                raise XTBloomValueError(f"unknown backend {backend!r}")
            self._requested = int(backend)
        self._device_id = -1 if device_id is None else int(device_id)
        self._cpu_threads = int(cpu_threads)
        if self._cpu_threads < 0:
            raise XTBloomValueError("cpu_threads must be nonnegative")
        if stream is not None:
            stream = int(stream)
            if stream <= 0:
                raise XTBloomValueError("stream must be a positive CUstream handle")
            if self._requested == library.BACKEND_CPU:
                raise XTBloomValueError(
                    "a native GPU stream cannot be attached to the CPU backend"
                )
        self._stream = stream
        self._handle = None
        self._backend: int | None = None
        self._finalizer: weakref.finalize | None = None
        self._plans: weakref.WeakSet[library.Plan] = weakref.WeakSet()
        # Tracks whether the native context holds a whole-batch converged
        # checkpoint that a later strict WARM request on the same context may
        # consume. Mirrors the native readiness gate: set only when every
        # system of the previous high-level call fully converged, cleared on
        # close (the native checkpoint dies with the context).
        self._warm_ready = False

    def _create(self) -> ctypes.c_void_p:
        if self._handle is not None:
            return self._handle
        library_instance = library.load_library()
        options = library.ContextOptions()
        library._check_init(
            "xtbloom_context_options_init",
            library_instance.xtbloom_context_options_init(
                ctypes.byref(options), ctypes.sizeof(options)
            ),
        )
        options.backend = self._requested
        options.device_id = self._device_id
        options.cpu_threads = self._cpu_threads
        options.stream = ctypes.c_void_p(self._stream or 0)
        handle = ctypes.c_void_p()
        status = library_instance.xtbloom_context_create(
            ctypes.byref(options), ctypes.byref(handle)
        )
        if status == library.STATUS_BACKEND_UNAVAILABLE:
            raise XTBloomRuntimeError(
                f"backend unavailable: {library.get_last_error()}", status
            )
        if status != library.STATUS_SUCCESS:
            raise XTBloomRuntimeError(
                f"xtbloom_context_create failed with {library.status_string(status)}: "
                f"{library.get_last_error()}",
                status,
            )
        self._handle = handle
        self._backend = library_instance.xtbloom_context_get_backend(handle)
        # Keep both the CDLL and the native handle alive in the finalizer so
        # contexts are reclaimed even when users follow the concise examples
        # and do not call close() explicitly.
        self._finalizer = weakref.finalize(
            self, _destroy_native_context, library_instance, handle
        )
        return self._handle

    @property
    def backend(self) -> int:
        """The resolved backend (CPU or CUDA)."""
        self._create()
        assert self._backend is not None  # _create() always resolves the backend
        return self._backend

    @property
    def device_id(self) -> int:
        """The backend device id (``-1`` for CPU)."""
        handle = self._create()
        return int(library.load_library().xtbloom_context_get_device_id(handle))

    @property
    def stream(self) -> int | None:
        """The native ``CUstream`` handle attached to this context, if any.

        ``None`` means the context uses the CUDA legacy default stream; the
        DLPack consumer in :class:`ArrayBatch` translates that to the DLPack
        stream value ``1``.
        """
        return self._stream

    def close(self) -> None:
        """Release the native context and its persistent resources."""
        if self._handle is not None:
            for plan in list(self._plans):
                plan.destroy()
            if self._finalizer is not None and self._finalizer.alive:
                self._finalizer()
            self._handle = None
            self._backend = None
            self._finalizer = None
            # A new native context created later starts without a checkpoint.
            self._warm_ready = False

    def create_plan(
        self, batch: library.Batch, options: library.ComputeOptions
    ) -> library.Plan:
        """Create a fixed-topology plan bound to this context.

        The low-level ``batch`` descriptor fixes the immutable topology; the
        returned plan reuses topology and workspace across repeated
        ``plan.compute`` calls (geometry may change). The plan must be
        destroyed before the context.
        """
        handle = self._create()
        plan = library.Plan(handle, batch, options, self)
        self._plans.add(plan)
        return plan

    def __enter__(self) -> Context:  # noqa: PYI034 - Python 3.10 lacks typing.Self
        """Create the native context and return this owner."""
        self._create()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the native context when leaving a ``with`` block."""
        self.close()


# --- low-level batch execution ----------------------------------------------------


@dataclass
class _ComputedBatch:
    """Raw native results and diagnostics for one ragged batch."""

    energies: np.ndarray
    forces: np.ndarray
    charges: np.ndarray
    point_charge_forces: np.ndarray | None
    dipole_moments: np.ndarray | None
    scc_iterations: np.ndarray
    scc_converged: np.ndarray
    per_system_status: np.ndarray
    result_flags: int
    atom_offsets: np.ndarray
    point_offsets: np.ndarray | None
    keepalive: list[np.ndarray]


def _pack_charge_responses(
    structures: Sequence[Structure],
) -> tuple[list[int], list[float], list[float]] | None:
    """Pack optional per-system response operators without penalizing common batches.

    Most callers do not provide periodic charge-response operators. Detect
    that case before constructing dense square matrices so ordinary inference
    remains linear in the input size. When any system has a response, systems
    without one receive explicit zero operators because the public C ABI uses
    one batch-wide descriptor set.
    """
    responses = [structure.charge_response for structure in structures]
    if not any(response is not None for response in responses):
        return None

    offsets = [0]
    shifts: list[float] = []
    matrices: list[float] = []
    for structure, response in zip(structures, responses, strict=True):
        atom_count = len(structure)
        if response is None:
            shifts.extend(0.0 for _ in range(atom_count))
            matrices.extend(0.0 for _ in range(atom_count * atom_count))
        else:
            if response.shifts.size != atom_count:
                raise XTBloomValueError(
                    "charge response shifts must match the atom count"
                )
            shifts.extend(float(value) for value in response.shifts)
            matrices.extend(float(value) for value in response.matrix.ravel())
        offsets.append(len(matrices))
    return offsets, shifts, matrices


# --- auto batch sizing --------------------------------------------------------

# Automatic sizing is deliberately conservative because CUDA workspace cost
# also depends on basis, spin, embedding, and per-system matrix extents. The
# estimate chooses an initial chunk size; allocation-failure retry below is the
# correctness backstop when a workload costs more than this atom proxy predicts.
_AUTO_BATCH_MEMORY_FRACTION = 0.5
_AUTO_BATCH_RESERVE_BYTES = 1_000_000_000
_AUTO_BATCH_BYTES_PER_ATOM = 400_000
_AUTO_BATCH_MAX_ATOMS = 65_536
_AUTO_BATCH_FALLBACK_MAX_ATOMS = 4_096


def _slice_by_total_atoms(
    structures: Sequence[Structure], max_total_atoms: int
) -> list[list[Structure]]:
    """Split *structures* into contiguous chunks of at most ``max_total_atoms``.

    A single system larger than the limit forms its own oversized chunk because
    systems are indivisible at the public C ABI. Thus the limit bounds grouped
    systems, while an oversized individual system is still attempted once.
    """
    if max_total_atoms < 1:
        raise XTBloomValueError("auto batch size must be a positive atom count")
    chunks: list[list[Structure]] = []
    current: list[Structure] = []
    current_atoms = 0
    for structure in structures:
        atoms = len(structure)
        if atoms > max_total_atoms:
            if current:
                chunks.append(current)
                current = []
                current_atoms = 0
            chunks.append([structure])
            continue
        if current and current_atoms + atoms > max_total_atoms:
            chunks.append(current)
            current = []
            current_atoms = 0
        current.append(structure)
        current_atoms += atoms
    if current:
        chunks.append(current)
    return chunks


def _merge_computed(
    computed_batches: Sequence[_ComputedBatch],
    structures: Sequence[Structure],
) -> _ComputedBatch:
    """Concatenate per-chunk results and rebase offsets into one public batch.

    Each chunk returns offsets relative to that chunk; the merged result must
    expose offsets relative to the complete structure list so ``BatchResult``
    slicing and indexing work uniformly.
    """
    point_batches = [
        batch.point_charge_forces
        for batch in computed_batches
        if batch.point_charge_forces is not None
    ]
    atom_offsets = [0]
    point_offsets = [0]
    for structure in structures:
        atom_offsets.append(atom_offsets[-1] + len(structure))
        point_offsets.append(
            point_offsets[-1] + len(structure.point_charges.charges)
            if structure.point_charges is not None
            else point_offsets[-1]
        )
    keepalive: list[np.ndarray] = []
    for batch in computed_batches:
        keepalive.extend(batch.keepalive)
    return _ComputedBatch(
        energies=np.concatenate([batch.energies for batch in computed_batches]),
        forces=np.concatenate([batch.forces for batch in computed_batches], axis=0),
        charges=np.concatenate([batch.charges for batch in computed_batches], axis=0),
        point_charge_forces=(
            np.concatenate(point_batches, axis=0) if point_batches else None
        ),
        dipole_moments=(
            np.concatenate([batch.dipole_moments for batch in computed_batches], axis=0)
            if any(batch.dipole_moments is not None for batch in computed_batches)
            else None
        ),
        scc_iterations=np.concatenate(
            [batch.scc_iterations for batch in computed_batches]
        ),
        scc_converged=np.concatenate(
            [batch.scc_converged for batch in computed_batches]
        ),
        per_system_status=np.concatenate(
            [batch.per_system_status for batch in computed_batches]
        ),
        result_flags=_merged_result_flags(computed_batches),
        atom_offsets=np.asarray(atom_offsets, dtype=np.int64),
        point_offsets=np.asarray(point_offsets, dtype=np.int64)
        if any(batch.point_offsets is not None for batch in computed_batches)
        else None,
        keepalive=keepalive,
    )


def _merged_result_flags(computed_batches: Sequence[_ComputedBatch]) -> int:
    """Preserve every batch-wide result qualifier produced by any chunk."""
    flags = 0
    for batch in computed_batches:
        flags |= batch.result_flags
    return flags


def _split_chunk_near_half(
    structures: Sequence[Structure],
) -> tuple[Sequence[Structure], Sequence[Structure]]:
    """Split a multi-system chunk near half its total atoms, preserving order."""
    if len(structures) < 2:
        raise XTBloomValueError("cannot split an indivisible batch chunk")
    target = sum(len(structure) for structure in structures) / 2
    cumulative = 0
    split = 1
    for index, structure in enumerate(structures[:-1], start=1):
        cumulative += len(structure)
        split = index
        if cumulative >= target:
            break
    return structures[:split], structures[split:]


def _resolve_auto_batch_limit(
    context: Context,
    structures: Sequence[Structure],
) -> int:
    """Choose a fresh, bounded atom proxy for one automatic compute chunk.

    Free memory is queried for every CUDA call so another process or an earlier
    xTBloom context cannot leave a stale cached limit. The estimate intentionally
    reserves half the reported free memory plus a fixed safety allowance. It is
    only an initial grouping heuristic: workload-specific allocation failures
    are handled by splitting multi-system chunks in :meth:`BatchCalculator.compute`.
    """
    total_atoms = sum(len(structure) for structure in structures)
    if int(context.backend) != library.BACKEND_CUDA:
        return min(total_atoms, _AUTO_BATCH_FALLBACK_MAX_ATOMS)

    memory = library.device_memory_info(int(context.device_id))
    if memory is None:
        return min(total_atoms, _AUTO_BATCH_FALLBACK_MAX_ATOMS)

    free_bytes, _ = memory
    budget = max(
        0,
        int(free_bytes * _AUTO_BATCH_MEMORY_FRACTION) - _AUTO_BATCH_RESERVE_BYTES,
    )
    estimated_limit = max(1, budget // _AUTO_BATCH_BYTES_PER_ATOM)
    return min(total_atoms, estimated_limit, _AUTO_BATCH_MAX_ATOMS)


def _compute_batch(
    context: Context,
    structures: Sequence[Structure],
    *,
    model: int,
    max_scc_iterations: int,
    charge_tolerance: float,
    energy_tolerance: float,
    electronic_temperature: float,
    flags: int,
    warm_start: bool = False,
) -> _ComputedBatch:
    """Populate descriptors and run one synchronous ``xtbloom_compute`` call.

    ``warm_start`` seeds SCC from the previous fully converged compatible
    electronic state retained on ``context`` (native ABI-v2 ``SCC_START_WARM``).
    The first call on a context, or any call whose topology or compute policy
    does not match the predecessor, falls back to an independent FRESH solve so
    automatic warm start stays transparent to the caller.
    """
    context._create()

    # --- assemble the ragged inputs -------------------------------------------
    atom_offsets = [0]
    atomic_numbers: list[int] = []
    positions: list[float] = []
    molecular_charges: list[float] = []
    unpaired_electrons: list[int] = []
    spin_channels: list[int] = []
    point_offsets = [0]
    point_positions: list[float] = []
    point_values: list[float] = []
    point_gammas: list[float] = []
    packed_responses = _pack_charge_responses(structures)
    keepalive: list = []
    field_payload: list[int] = []
    field_descriptors: list[object] = []

    for structure in structures:
        atomic_numbers.extend(int(number) for number in structure.numbers)
        positions.extend(float(value) for value in structure.positions.ravel())
        molecular_charges.append(structure.charge)
        unpaired_electrons.append(structure.uhf)
        spin_channels.append(structure.spin_channels)
        if structure.point_charges is not None:
            points = structure.point_charges
            point_positions.extend(float(value) for value in points.positions.ravel())
            point_values.extend(float(value) for value in points.charges)
            point_gammas.extend(float(value) for value in points.gammas)
        if structure.efield is not None:
            system_index = len(molecular_charges) - 1
            descriptor = library.Interaction()
            descriptor.type = library.INTERACTION_ELECTRIC_FIELD
            descriptor.flags = 0
            descriptor.system_index = system_index
            descriptor.payload_offset = len(field_payload)
            descriptor.payload_size = 32
            field_descriptors.append(descriptor)
            field_payload.extend(struct.pack("<i", 1))  # block_version
            field_payload.extend(struct.pack("<i", 0))  # reserved
            field_payload.extend(struct.pack("<3d", *structure.efield))
        atom_offsets.append(len(atomic_numbers))
        point_offsets.append(len(point_values))

    total_atoms = len(atomic_numbers)
    total_points = len(point_values)

    # --- bind descriptors ------------------------------------------------------
    library_instance = library.load_library()
    batch = library.Batch()
    library._check_init(
        "xtbloom_batch_init",
        library_instance.xtbloom_batch_init(ctypes.byref(batch), ctypes.sizeof(batch)),
    )
    batch.batch_size = len(structures)
    batch.total_atoms = total_atoms
    batch.total_point_charges = total_points
    batch.total_charge_response_elements = (
        len(packed_responses[2]) if packed_responses is not None else 0
    )

    def bind(
        descriptor_name: str,
        values: Sequence[int | float],
        dtype: npt.DTypeLike,
    ) -> None:
        if not values:
            setattr(
                batch,
                descriptor_name,
                library.ConstBuffer(None, 0, library.MEMORY_HOST, 0),
            )
            return
        owner = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        keepalive.append(owner)
        setattr(
            batch,
            descriptor_name,
            library.ConstBuffer(
                ctypes.cast(owner.ctypes.data, ctypes.c_void_p),
                owner.nbytes,
                library.MEMORY_HOST,
                0,
            ),
        )

    bind("atom_offsets", atom_offsets, np.int64)
    bind("atomic_numbers", atomic_numbers, np.int32)
    bind("positions", positions, np.float64)
    bind("molecular_charges", molecular_charges, np.float64)
    bind("unpaired_electrons", unpaired_electrons, np.int32)
    bind("spin_channels", spin_channels, np.int32)
    if total_points:
        bind("point_charge_offsets", point_offsets, np.int64)
        bind("point_charge_positions", point_positions, np.float64)
        bind("point_charge_values", point_values, np.float64)
        bind("point_charge_gammas", point_gammas, np.float64)
    if packed_responses is not None:
        response_offsets, response_shifts, response_matrix = packed_responses
        bind("atomic_potential_shifts", response_shifts, np.float64)
        bind("charge_response_offsets", response_offsets, np.int64)
        bind("charge_response_matrix", response_matrix, np.float64)
    if field_descriptors:
        batch.total_interactions = len(field_descriptors)
        descriptor_owner = (library.Interaction * len(field_descriptors))(
            *field_descriptors
        )
        keepalive.append(descriptor_owner)
        batch.interaction_descriptors = library.ConstBuffer(
            ctypes.cast(descriptor_owner, ctypes.c_void_p),
            ctypes.sizeof(descriptor_owner),
            library.MEMORY_HOST,
            0,
        )
        payload_owner = np.frombuffer(bytes(field_payload), dtype=np.uint8)
        keepalive.append(payload_owner)
        batch.interaction_payload = library.ConstBuffer(
            ctypes.cast(payload_owner.ctypes.data, ctypes.c_void_p),
            payload_owner.nbytes,
            library.MEMORY_HOST,
            0,
        )

    # --- compute options -------------------------------------------------------
    options = library.ComputeOptions()
    library._check_init(
        "xtbloom_compute_options_init",
        library_instance.xtbloom_compute_options_init(
            ctypes.byref(options), ctypes.sizeof(options)
        ),
    )
    # High-level calculators default to reproducible independent SCC solves;
    # automatic warm start is opt-in. When enabled and the native context holds
    # a fully converged compatible checkpoint, request SCC_START_WARM so the
    # previous converged electronic state seeds the new SCC run (geometry is
    # not part of the native identity, which is why this is ideal for dynamics).
    options.scc_start_mode = (
        library.SCC_START_WARM
        if warm_start and context._warm_ready
        else library.SCC_START_FRESH
    )
    options.model = model
    options.flags = flags
    options.max_scc_iterations = int(max_scc_iterations)
    options.charge_tolerance = float(charge_tolerance)
    options.energy_tolerance = float(energy_tolerance)
    options.electronic_temperature = (
        float(electronic_temperature) * library.KELVIN_TO_HARTREE
    )

    # --- result buffers --------------------------------------------------------
    result = library.BatchResult()
    library._check_init(
        "xtbloom_batch_result_init",
        library_instance.xtbloom_batch_result_init(
            ctypes.byref(result), ctypes.sizeof(result)
        ),
    )

    nsystems = len(structures)
    energies = np.empty(nsystems, dtype=np.float64)
    forces = np.empty((total_atoms, 3), dtype=np.float64)
    charges = np.empty(total_atoms, dtype=np.float64)
    point_charge_forces = (
        np.empty((total_points, 3), dtype=np.float64) if total_points else None
    )
    dipole_moments = (
        np.empty((nsystems, 3), dtype=np.float64)
        if bool(flags & library.COMPUTE_DIPOLE_MOMENTS)
        else None
    )
    scc_iterations = np.empty(nsystems, dtype=np.int32)
    scc_converged = np.empty(nsystems, dtype=np.uint8)
    per_system_status = np.empty(nsystems, dtype=np.int32)

    def bind_output(
        buffer_field: str, owner: np.ndarray | None, requested: bool
    ) -> None:
        if not requested or owner is None:
            setattr(
                result, buffer_field, library.Buffer(None, 0, library.MEMORY_HOST, 0)
            )
            return
        keepalive.append(owner)
        setattr(
            result,
            buffer_field,
            library.Buffer(
                ctypes.cast(owner.ctypes.data, ctypes.c_void_p),
                owner.nbytes,
                library.MEMORY_HOST,
                0,
            ),
        )

    bind_output("energies", energies, bool(flags & library.COMPUTE_ENERGY))
    bind_output("forces", forces, bool(flags & library.COMPUTE_FORCES))
    bind_output("atomic_charges", charges, bool(flags & library.COMPUTE_ATOMIC_CHARGES))
    bind_output(
        "point_charge_forces",
        point_charge_forces,
        bool(flags & library.COMPUTE_POINT_CHARGE_FORCES),
    )
    bind_output(
        "dipole_moments",
        dipole_moments,
        bool(flags & library.COMPUTE_DIPOLE_MOMENTS),
    )
    bind_output("scc_iterations", scc_iterations, True)
    bind_output("scc_converged", scc_converged, True)
    bind_output("per_system_status", per_system_status, True)

    try:
        library.compute_checked(context._create(), batch, options, result)
    except XTBloomRuntimeError as error:
        # A strict WARM request with no compatible fully converged predecessor
        # is rejected by the native gate before any caller output is modified
        # and without side effects, so one independent FRESH retry is safe.
        # This keeps automatic warm start transparent when the identity changed
        # (e.g. charge, spin, tolerances, topology) or the previous call did
        # not fully converge.
        if (
            options.scc_start_mode != library.SCC_START_WARM
            or error.status != library.STATUS_INVALID_ARGUMENT
        ):
            raise
        options.scc_start_mode = library.SCC_START_FRESH
        library.compute_checked(context._create(), batch, options, result)

    # Mirror the native whole-batch readiness gate: only a call in which every
    # system fully converged leaves a consumable warm checkpoint on the context.
    context._warm_ready = all(
        int(status) == library.STATUS_SUCCESS and int(converged) == 1
        for status, converged in zip(per_system_status, scc_converged, strict=True)
    )

    return _ComputedBatch(
        energies=energies,
        forces=forces,
        charges=charges,
        point_charge_forces=point_charge_forces,
        dipole_moments=dipole_moments,
        scc_iterations=scc_iterations,
        scc_converged=scc_converged,
        per_system_status=per_system_status,
        result_flags=int(result.flags),
        atom_offsets=np.asarray(atom_offsets, dtype=np.int64),
        point_offsets=np.asarray(point_offsets, dtype=np.int64)
        if total_points
        else None,
        keepalive=keepalive,
    )


def _raise_on_failure(computed: _ComputedBatch) -> None:
    """Raise when any system failed SCC or the eigensolver.

    The C ABI fills failed systems' floating-point results with NaNs, so a
    partially failed batch is treated as an error by the high-level interface.
    """
    failed = []
    for index, (status, converged) in enumerate(
        zip(computed.per_system_status, computed.scc_converged, strict=True)
    ):
        if int(status) != library.STATUS_SUCCESS or int(converged) != 1:
            failed.append(
                f"system {index}: {library.status_string(int(status))}, "
                f"scc_converged={int(converged)}, "
                f"iterations={int(computed.scc_iterations[index])}"
            )
    if failed:
        raise XTBloomRuntimeError(
            "xTBloom batch inference produced failed systems: " + "; ".join(failed)
        )


# --- results ----------------------------------------------------------------------


class Result:
    """Single-system results container, similar to ``tblite.interface.Result``."""

    _getter: ClassVar[builtins.dict[str, Callable[[Result], object]]] = {
        "energy": lambda self: self.energy,
        "energies": lambda self: np.asarray([self.energy]),
        "forces": lambda self: self.forces,
        # The public C ABI returns force = -dE/dR, whereas tblite-style
        # ``gradient`` means +dE/dR.
        "gradient": lambda self: -self.forces,
        "charges": lambda self: self.charges,
        "point_charge_forces": lambda self: self.point_charge_forces,
        "dipole_moments": lambda self: self.dipole_moments,
        "scc_iterations": lambda self: self.scc_iterations,
        "scc_converged": lambda self: self.scc_converged,
        "scc_status": lambda self: self.scc_status,
        "natoms": lambda self: self.natoms,
    }

    def __init__(self, computed: _ComputedBatch, index: int = 0) -> None:
        begin = int(computed.atom_offsets[index])
        end = int(computed.atom_offsets[index + 1])
        self._natoms = end - begin
        self.energy = float(computed.energies[index])
        self.forces = np.array(computed.forces[begin:end], copy=True)
        self.charges = np.array(computed.charges[begin:end], copy=True)
        self.point_charge_forces = None
        if computed.point_charge_forces is not None:
            if computed.point_offsets is not None:
                point_begin = int(computed.point_offsets[index])
                point_end = int(computed.point_offsets[index + 1])
            else:
                point_begin, point_end = 0, len(computed.point_charge_forces)
            self.point_charge_forces = np.array(
                computed.point_charge_forces[point_begin:point_end], copy=True
            )
        self.dipole_moments = (
            np.array(computed.dipole_moments[index], copy=True)
            if computed.dipole_moments is not None
            else None
        )
        self.scc_iterations = int(computed.scc_iterations[index])
        self.scc_converged = bool(computed.scc_converged[index])
        self.scc_status = int(computed.per_system_status[index])

    @property
    def natoms(self) -> int:
        """Return the number of atoms represented by this result."""
        return self._natoms

    def get(self, attribute: str) -> object:
        """Return a requested quantity by name.

        Available keys: ``energy``, ``energies``, ``forces``, ``gradient``
        (the negative of forces), ``charges``, ``point_charge_forces``,
        ``dipole_moments``, ``scc_iterations``,
        ``scc_converged``, ``scc_status``, and ``natoms``.
        """
        if attribute not in self._getter:
            raise XTBloomValueError(
                f"attribute {attribute!r} is not available in this result"
            )
        return self._getter[attribute](self)

    def __getitem__(self, key: str) -> object:
        """Return a requested result quantity by key."""
        return self.get(key)

    def dict(self) -> builtins.dict[str, object]:
        """Return all available quantities in a new mapping."""
        return {key: self.get(key) for key in self._getter}


class BatchResult:
    """Multi-system results container for :class:`BatchCalculator`."""

    def __init__(
        self, computed: _ComputedBatch, structures: Sequence[Structure]
    ) -> None:
        self.energies = np.array(computed.energies, copy=True)
        self.forces = np.array(computed.forces, copy=True)
        self.charges = np.array(computed.charges, copy=True)
        self.point_charge_forces = (
            np.array(computed.point_charge_forces, copy=True)
            if computed.point_charge_forces is not None
            else None
        )
        self.dipole_moments = (
            np.array(computed.dipole_moments, copy=True)
            if computed.dipole_moments is not None
            else None
        )
        self.scc_iterations = np.array(computed.scc_iterations, copy=True)
        self.scc_converged = np.array(computed.scc_converged, copy=True)
        self.per_system_status = np.array(computed.per_system_status, copy=True)
        self._structures = list(structures)
        self._atom_offsets = np.array(computed.atom_offsets, copy=True)
        self._point_offsets = (
            np.array(computed.point_offsets, copy=True)
            if computed.point_offsets is not None
            else None
        )

    def __len__(self) -> int:
        """Return the number of systems in this batch result."""
        return len(self._structures)

    def __getitem__(self, index: int) -> Result:
        """Return a detached single-system result by index."""
        try:
            index = operator.index(index)
        except TypeError:
            raise TypeError("batch result indices must be integers") from None
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("batch result index out of range")
        return Result(
            _ComputedBatch(
                energies=self.energies,
                forces=self.forces,
                charges=self.charges,
                point_charge_forces=self.point_charge_forces,
                dipole_moments=self.dipole_moments,
                scc_iterations=self.scc_iterations,
                scc_converged=self.scc_converged,
                per_system_status=self.per_system_status,
                result_flags=0,
                atom_offsets=self._atom_offsets,
                point_offsets=self._point_offsets,
                keepalive=[],
            ),
            index=index,
        )

    @property
    def failed_indices(self) -> np.ndarray:
        """Indices whose SCC/eigensolver status is not successful."""
        return np.flatnonzero(
            (self.per_system_status != library.STATUS_SUCCESS)
            | (self.scc_converged != 1)
        )

    def raise_for_status(self) -> None:
        """Raise a combined exception while retaining peer-local results."""
        failed = [
            f"system {int(index)}: "
            f"{library.status_string(int(self.per_system_status[index]))}, "
            f"scc_converged={int(self.scc_converged[index])}, "
            f"iterations={int(self.scc_iterations[index])}"
            for index in self.failed_indices
        ]
        if failed:
            raise XTBloomRuntimeError(
                "xTBloom batch inference produced failed systems: " + "; ".join(failed)
            )

    def get(self, attribute: str) -> object:
        """Return a batch array by name, including optional dipole moments."""
        names = {
            "energies": self.energies,
            "forces": self.forces,
            "charges": self.charges,
            "point_charge_forces": self.point_charge_forces,
            "dipole_moments": self.dipole_moments,
            "scc_iterations": self.scc_iterations,
            "scc_converged": self.scc_converged,
            "per_system_status": self.per_system_status,
        }
        if attribute not in names:
            raise XTBloomValueError(
                f"attribute {attribute!r} is not available in this result"
            )
        return names[attribute]


# --- calculators -------------------------------------------------------------------


def _validated_compute_setting(attribute: str, value: object) -> int | float:
    """Validate one compute setting and return its normalized scalar value."""
    if attribute == "max_scc_iterations":
        candidate = _as_integer(attribute, value)
        if candidate <= 0:
            raise XTBloomValueError("max_scc_iterations must be positive")
        return candidate
    if attribute in ("charge_tolerance", "energy_tolerance"):
        candidate = float(typing.cast("SupportsFloat | SupportsIndex", value))
        if not math.isfinite(candidate) or candidate <= 0.0:
            raise XTBloomValueError(f"{attribute} must be finite and positive")
        return candidate
    if attribute == "electronic_temperature":
        candidate = float(typing.cast("SupportsFloat | SupportsIndex", value))
        if not math.isfinite(candidate) or candidate < 0.0:
            raise XTBloomValueError(
                "electronic_temperature must be finite and nonnegative"
            )
        return candidate
    raise XTBloomValueError(f"unsupported calculator setting {attribute!r}")


class _ComputeSettings:
    __slots__ = (
        "charge_tolerance",
        "electronic_temperature",
        "energy_tolerance",
        "max_scc_iterations",
        "model",
    )

    model: int
    max_scc_iterations: int
    charge_tolerance: float
    energy_tolerance: float
    electronic_temperature: float

    def __init__(
        self,
        model: int,
        max_scc_iterations: int,
        charge_tolerance: float,
        energy_tolerance: float,
        electronic_temperature: float,
    ) -> None:
        self.model = model
        self.set("max_scc_iterations", max_scc_iterations)
        self.set("charge_tolerance", charge_tolerance)
        self.set("energy_tolerance", energy_tolerance)
        self.set("electronic_temperature", electronic_temperature)

    def set(self, attribute: str, value: object) -> None:
        """Validate and transactionally update one public compute option."""
        setattr(self, attribute, _validated_compute_setting(attribute, value))


def _resolve_method(method: str) -> int:
    if not method:
        raise XTBloomValueError(
            "a method must be provided (only GFN2-xTB is currently supported)"
        )
    try:
        return _SUPPORTED_METHODS[method]
    except KeyError:
        if method in ("GFN1-xTB", "GFN1"):
            raise XTBloomNotSupportedError(
                "GFN1-xTB is reserved by the xTBloom ABI but is not implemented yet"
            ) from None
        raise XTBloomValueError(f"unknown method {method!r}") from None


class Calculator(Structure):
    """Single-point GFN2-xTB calculator for one structure (tblite-like API).

    Example
    -------
    >>> import numpy as np
    >>> from xtbloom.interface import Calculator
    >>> calc = Calculator(
    ...     "GFN2-xTB",
    ...     numbers=np.array([8, 1, 1]),
    ...     positions=np.array([
    ...         [+0.00000000000000, +0.00000000000000, -0.73578586109551],
    ...         [+1.44183152868459, +0.00000000000000, +0.36789293054775],
    ...         [-1.44183152868459, +0.00000000000000, +0.36789293054775],
    ...     ]),
    ... )
    >>> res = calc.singlepoint()
    >>> res.get("energy")  # Hartree

    ``warm_start=True`` seeds each SCC solve from the previous fully converged
    compatible electronic state retained on the same context (native ABI-v2
    ``SCC_START_WARM``). Geometry is not part of the native identity, so a
    dynamics or optimization loop reusing one ``Calculator`` reconverges from
    the previous step's state; the first call and any identity change fall back
    to an independent FRESH solve. Results converge to the same electronic
    state within SCC tolerance but may differ in the last converged digits.
    """

    def __init__(
        self,
        method: str,
        numbers: np.ndarray | list[int] | Sequence[str],
        positions: np.ndarray,
        charge: float = 0.0,
        uhf: int | None = None,
        multiplicity: int | None = None,
        spin_channels: int | None = None,
        point_charges: PointCharge | None = None,
        charge_response: ChargeResponse | None = None,
        efield: np.ndarray | list[float] | None = None,
        *,
        backend: str | int = "auto",
        device_id: int | None = None,
        cpu_threads: int = 1,
        max_scc_iterations: int = 250,
        charge_tolerance: float = 1.0e-6,
        energy_tolerance: float = 1.0e-8,
        electronic_temperature: float = 300.0,
        warm_start: bool = False,
    ) -> None:
        Structure.__init__(
            self,
            numbers,
            positions,
            charge,
            uhf=uhf,
            multiplicity=multiplicity,
            spin_channels=spin_channels,
            point_charges=point_charges,
            charge_response=charge_response,
            efield=efield,
        )
        self._model = _resolve_method(method)
        self._settings = _ComputeSettings(
            self._model,
            max_scc_iterations,
            charge_tolerance,
            energy_tolerance,
            electronic_temperature,
        )
        self._context = Context(backend, device_id, cpu_threads)
        self._method = method
        self._warm_start = bool(warm_start)

    @property
    def backend(self) -> int:
        """The resolved execution backend of this calculator."""
        return self._context.backend

    @property
    def method(self) -> str:
        """Return the configured tight-binding method name."""
        return self._method

    def update(
        self,
        positions: np.ndarray | None = None,
        charge: float | None = None,
        uhf: int | None = None,
        multiplicity: int | None = None,
        spin_channels: int | None = None,
    ) -> None:
        """Update the structure geometry and/or electronic state in place."""
        Structure.update(
            self,
            positions=positions,
            charge=charge,
            uhf=uhf,
            multiplicity=multiplicity,
            spin_channels=spin_channels,
        )

    def set(self, attribute: str, value: object) -> None:
        """Update a compute setting by name.

        Supported settings are ``max_scc_iterations``, ``charge_tolerance``,
        ``energy_tolerance``, and ``electronic_temperature`` (kelvin).
        """
        self._settings.set(attribute, value)

    def singlepoint(self) -> Result:
        """Perform a single-point calculation and return a :class:`Result`."""
        flags = (
            library.COMPUTE_ENERGY
            | library.COMPUTE_FORCES
            | library.COMPUTE_ATOMIC_CHARGES
        )
        if self.point_charges is not None:
            flags |= library.COMPUTE_POINT_CHARGE_FORCES
        if self.efield is not None:
            flags |= library.COMPUTE_DIPOLE_MOMENTS
        computed = _compute_batch(
            self._context,
            [self],
            model=self._settings.model,
            max_scc_iterations=self._settings.max_scc_iterations,
            charge_tolerance=self._settings.charge_tolerance,
            energy_tolerance=self._settings.energy_tolerance,
            electronic_temperature=self._settings.electronic_temperature,
            flags=flags,
            warm_start=self._warm_start,
        )
        _raise_on_failure(computed)
        return Result(computed, index=0)

    def close(self) -> None:
        """Release this calculator's native context."""
        self._context.close()

    def __enter__(self) -> Calculator:  # noqa: PYI034 - Python 3.10 lacks typing.Self
        """Return this calculator for use in a ``with`` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release native resources when leaving a ``with`` block."""
        self.close()


class BatchCalculator:
    """Batched GFN2-xTB calculator over many structures in one C call.

    The C API describes a ragged batch with flat arrays and offsets, so all
    systems are solved together while keeping per-system convergence state.
    ``warm_start=True`` reuses the previous fully converged state only when
    each logical batch remains one native call; it cannot be combined with
    ``auto_batch_size`` because a context retains just its latest whole-batch
    checkpoint rather than one checkpoint per chunk.
    """

    def __init__(
        self,
        structures: Sequence[Structure],
        method: str = "GFN2-xTB",
        *,
        backend: str | int = "auto",
        device_id: int | None = None,
        cpu_threads: int = 1,
        max_scc_iterations: int = 250,
        charge_tolerance: float = 1.0e-6,
        energy_tolerance: float = 1.0e-8,
        electronic_temperature: float = 300.0,
        warm_start: bool = False,
    ) -> None:
        if not structures:
            raise XTBloomValueError("a batch needs at least one structure")
        self._structures = list(structures)
        self._settings = _ComputeSettings(
            _resolve_method(method),
            max_scc_iterations,
            charge_tolerance,
            energy_tolerance,
            electronic_temperature,
        )
        self._context = Context(backend, device_id, cpu_threads)
        self._warm_start = bool(warm_start)

    def __len__(self) -> int:
        """Return the number of structures in this calculator."""
        return len(self._structures)

    @property
    def backend(self) -> int:
        """Return the resolved execution backend."""
        return self._context.backend

    def set(self, attribute: str, value: object) -> None:
        """Validate and update a compute setting by name."""
        self._settings.set(attribute, value)

    def compute(
        self,
        *,
        raise_on_failure: bool = False,
        auto_batch_size: bool | int | None = None,
    ) -> BatchResult:
        """Run the batch while preserving successful peers.

        By default the result is returned even when individual systems fail;
        their floating-point slices contain NaNs and diagnostics identify the
        failed peers.  Set ``raise_on_failure=True`` or call
        :meth:`BatchResult.raise_for_status` for strict behavior.

        ``auto_batch_size`` controls automatic slicing of one large batch into
        several ``xtbloom_compute`` calls. ``None`` or ``False`` preserves the
        historical single-call behavior. An integer is a target maximum total
        atom count per chunk (``1`` forces one system per call). ``True`` picks
        a conservative target from the CUDA device's current free memory, or a
        fixed fallback when memory cannot be queried, and retries native
        allocation failures by splitting multi-system chunks. A system larger
        than the target remains indivisible and is attempted by itself.

        Slicing preserves system order and peer-local failures. CPU results are
        bit-identical to an unsliced run. CUDA chunking can change eigensolver
        bucket composition, so CUDA results should be compared with the same
        tolerances used for ordinary backend conformance.

        ``auto_batch_size`` cannot be combined with ``warm_start=True``. The
        native context owns one whole-batch checkpoint, so sharing it across
        chunks could seed a system from a different chunk rather than from the
        corresponding system in the preceding logical batch.
        """
        if self._warm_start and auto_batch_size not in (None, False):
            raise XTBloomValueError(
                "warm_start=True cannot be combined with auto_batch_size; "
                "use one native batch or disable warm start"
            )

        base_flags = (
            library.COMPUTE_ENERGY
            | library.COMPUTE_FORCES
            | library.COMPUTE_ATOMIC_CHARGES
        )
        if any(structure.efield is not None for structure in self._structures):
            # Request a uniform result shape for every auto-sized chunk. A
            # logical mixed field/plain batch must not produce some chunks
            # with dipoles and others without them.
            base_flags |= library.COMPUTE_DIPOLE_MOMENTS

        def run_once(structures: Sequence[Structure]) -> _ComputedBatch:
            # Output descriptors are batch-local. In particular, requesting a
            # point-charge output for a zero-point chunk is inconsistent with
            # the native CUDA publication plan even when another chunk has
            # point charges.
            flags = base_flags
            if any(structure.point_charges is not None for structure in structures):
                flags |= library.COMPUTE_POINT_CHARGE_FORCES
            return _compute_batch(
                self._context,
                structures,
                model=self._settings.model,
                max_scc_iterations=self._settings.max_scc_iterations,
                charge_tolerance=self._settings.charge_tolerance,
                energy_tolerance=self._settings.energy_tolerance,
                electronic_temperature=self._settings.electronic_temperature,
                flags=flags,
                warm_start=self._warm_start,
            )

        if auto_batch_size is None or auto_batch_size is False:
            chunks: list[Sequence[Structure]] = [self._structures]
        elif auto_batch_size is True:
            limit = _resolve_auto_batch_limit(self._context, self._structures)
            chunks = _slice_by_total_atoms(self._structures, limit)
        else:
            limit = _as_integer("auto_batch_size", auto_batch_size)
            if limit <= 0:
                raise XTBloomValueError("auto_batch_size must be a positive integer")
            chunks = _slice_by_total_atoms(self._structures, limit)

        def run_auto_chunk(chunk: Sequence[Structure]) -> list[_ComputedBatch]:
            """Retry only recoverable native allocation failures at smaller sizes."""
            try:
                return [run_once(chunk)]
            except XTBloomRuntimeError as error:
                if (
                    auto_batch_size is not True
                    or error.status != library.STATUS_ALLOCATION_FAILED
                    or len(chunk) == 1
                ):
                    raise
                left, right = _split_chunk_near_half(chunk)
                return [*run_auto_chunk(left), *run_auto_chunk(right)]

        computed_batches = [
            computed for chunk in chunks for computed in run_auto_chunk(chunk)
        ]
        computed = (
            computed_batches[0]
            if len(computed_batches) == 1
            else _merge_computed(computed_batches, self._structures)
        )
        batch_result = BatchResult(computed, self._structures)
        if raise_on_failure:
            batch_result.raise_for_status()
        return batch_result

    def close(self) -> None:
        """Release this batch calculator's native context."""
        self._context.close()

    def __enter__(self) -> BatchCalculator:  # noqa: PYI034 - Python 3.10 lacks typing.Self
        """Return this calculator for use in a ``with`` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release native resources when leaving a ``with`` block."""
        self.close()


# --- packed Array API / DLPack batch execution --------------------------------


def _probe_shape(array: object) -> tuple[int, ...]:
    """Read the logical shape from an eager Array API array.

    Only the metadata is touched; nothing is evaluated or copied.  The shape
    is used to derive the ragged-batch extents so the full request can be
    validated before any DLPack capsule is consumed.
    """
    if _array_adapter.is_lazy(array):
        raise XTBloomNotSupportedError(
            f"{_array_adapter.backend_name(array)} tracer/lazy objects cannot be "
            "used with xTBloom; pass a concrete eager array"
        )
    shape = getattr(array, "shape", None)
    if shape is None:
        raise XTBloomValueError(
            f"{_array_adapter.backend_name(array)} does not expose .shape; "
            "xTBloom requires eager Array API arrays"
        )
    return tuple(int(value) for value in shape)


# Optional descriptor groups that must be supplied all-or-nothing, matching the
# C ABI's single batch-wide descriptor sets.
_POINT_CHARGE_FIELDS = (
    "point_charge_offsets",
    "point_charge_positions",
    "point_charge_values",
    "point_charge_gammas",
)
_CHARGE_RESPONSE_FIELDS = (
    "atomic_potential_shifts",
    "charge_response_offsets",
    "charge_response_matrix",
)


class ArrayBatch:
    """Packed ragged-batch inference over Array API/DLPack arrays.

    This is the zero-copy entry point of the Python interface: every
    positional descriptor takes a dense, single-device array implementing the
    Array API ``__dlpack__``/``__dlpack_device__`` producer protocols (NumPy,
    CuPy, JAX eager arrays, and PyTorch tensors).  xTBloom never imports those
    libraries; the caller's arrays are consumed through DLPack and their
    buffers are bound directly to the C ABI descriptors:

    * Host (CPU) arrays become ``XTBLOOM_MEMORY_HOST`` descriptors.
    * CUDA device arrays become ``XTBLOOM_MEMORY_CUDA_DEVICE`` descriptors and
      are executed by the CUDA backend without a host round trip.

    ``copy=False`` (the default) requires exact dtype, shape, and compact
    C-contiguous layout; anything else raises instead of silently copying.
    Set ``copy=True`` to ask the producer for a contiguous copy.  CUDA-managed
    memory, ROCm, and other device kinds are rejected with a precise error.
    Lazy/tracer objects (``jit``/``grad``/``vmap`` inputs, ``torch.compile``
    graphs) are rejected with a precise error.  :mod:`xtbloom.torch` is the
    only autograd entry point: it consumes eager tensors through this same
    DLPack path and deliberately supports only the positions gradient
    (``dE/dR = -F``).

    The existing :class:`Structure`/:class:`Calculator` host-numpy path is
    unchanged and remains the compatibility default.

    Parameters
    ----------
    atom_offsets : (nsystems + 1,) int64
        Ragged atom offsets; ``offsets[-1]`` is the total atom count.
    atomic_numbers : (natoms,) int32
        Concatenated atomic numbers.
    positions : (natoms, 3) float64
        Concatenated Cartesian positions in bohr.
    molecular_charges : (nsystems,) float64
        Total molecular charge of each system.
    unpaired_electrons : (nsystems,) int32
        Number of unpaired electrons of each system.
    spin_channels : (nsystems,) int32, optional
        Orbital channels (1 restricted / 2 unrestricted); defaults to all
        restricted ``1`` on a host buffer.
    point_charge_offsets, point_charge_positions, point_charge_values,
    point_charge_gammas : optional
        External point-charge group; must be supplied together.  Offsets are
        ``(nsystems + 1,)`` int64, positions ``(npoints, 3)`` float64, values
        and gammas ``(npoints,)`` float64.
    atomic_potential_shifts, charge_response_offsets, charge_response_matrix : optional
        Periodic charge-response ``b + A q`` group; must be supplied together.
        Shifts are ``(natoms,)`` float64, offsets ``(nsystems + 1,)`` int64,
        and the matrix packs all per-system ``A`` blocks row-major.
    copy : bool
        Allow a producer-side copy to pack a non-contiguous input. Dtypes must
        still match the C ABI exactly.
    backend, device_id, cpu_threads : optional
        Same context selection as :class:`Calculator`.
    stream : int, optional
        Raw ``CUstream`` handle the native context should use.  ``None`` uses
        the CUDA legacy default stream; CUDA producers receive DLPack stream
        value ``1`` in that case and the raw handle otherwise.
    """

    def __init__(
        self,
        atom_offsets: object,
        atomic_numbers: object,
        positions: object,
        molecular_charges: object,
        unpaired_electrons: object,
        spin_channels: object | None = None,
        point_charge_offsets: object | None = None,
        point_charge_positions: object | None = None,
        point_charge_values: object | None = None,
        point_charge_gammas: object | None = None,
        atomic_potential_shifts: object | None = None,
        charge_response_offsets: object | None = None,
        charge_response_matrix: object | None = None,
        *,
        copy: bool = False,
        backend: str | int = "auto",
        device_id: int | None = None,
        cpu_threads: int = 1,
        stream: int | None = None,
    ) -> None:
        if not (
            all(
                hasattr(array, "__dlpack__") and hasattr(array, "__dlpack_device__")
                for array in (
                    atom_offsets,
                    atomic_numbers,
                    positions,
                    molecular_charges,
                    unpaired_electrons,
                )
            )
        ):
            raise XTBloomValueError(
                "ArrayBatch requires arrays implementing the DLPack producer "
                "protocol (__dlpack__ and __dlpack_device__)"
            )
        _require_all_or_none("point charge", _POINT_CHARGE_FIELDS, locals())
        _require_all_or_none("charge response", _CHARGE_RESPONSE_FIELDS, locals())
        self._arrays: dict[str, object | None] = {
            "atom_offsets": atom_offsets,
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "molecular_charges": molecular_charges,
            "unpaired_electrons": unpaired_electrons,
            "spin_channels": spin_channels,
            "point_charge_offsets": point_charge_offsets,
            "point_charge_positions": point_charge_positions,
            "point_charge_values": point_charge_values,
            "point_charge_gammas": point_charge_gammas,
            "atomic_potential_shifts": atomic_potential_shifts,
            "charge_response_offsets": charge_response_offsets,
            "charge_response_matrix": charge_response_matrix,
        }
        self._copy = bool(copy)
        self._context = Context(backend, device_id, cpu_threads, stream)

    @property
    def backend(self) -> int:
        """The resolved execution backend of this batch."""
        return self._context.backend

    @property
    def context(self) -> Context:
        """The native context this batch computes with."""
        return self._context

    def compute(
        self,
        *,
        max_scc_iterations: int = 250,
        charge_tolerance: float = 1.0e-6,
        energy_tolerance: float = 1.0e-8,
        electronic_temperature: float = 300.0,
        compute_energy: bool = True,
        compute_forces: bool = True,
        compute_charges: bool = True,
        compute_point_charge_forces: bool | None = None,
        out: object | None = None,
        result_memory: str = "host",
    ) -> ArrayBatchResult:
        """Run one synchronous inference and return an :class:`ArrayBatchResult`.

        Per-system SCC/eigensolver failures are data-level results, exactly as
        in :meth:`BatchCalculator.compute`: peers stay independent and a
        failed system's floating-point slices are NaNs.

        ``out`` may map output names to writable, correctly-typed arrays of
        any supported backend (NumPy, CuPy, or PyTorch); xTBloom writes
        directly into them instead of allocating host results.  JAX arrays are
        never mutated and are rejected as mutable outputs.  Names:
        ``energies``, ``forces``, ``charges`` (alias ``atomic_charges``),
        ``point_charge_forces``, ``scc_iterations``, ``scc_converged``, and
        ``per_system_status``.

        ``result_memory`` selects how outputs without an ``out=`` buffer are
        allocated.  ``"host"`` (the default) keeps the historical behavior:
        fresh host NumPy arrays.  ``"cuda"`` requires the resolved CUDA
        backend and allocates an xTBloom-owned device arena on the context
        device; every xTBloom-owned output is then returned as a
        :class:`xtbloom._dlpack.DLPackResultBuffer` producer that
        ``torch.from_dlpack``/``cupy.from_dlpack``/``jax.dlpack.from_dlpack``
        can import without a host copy.  Caller-supplied ``out=`` buffers
        always take precedence, so mixed caller-owned and xTBloom-owned outputs
        are allowed.  Device-resident ``per_system_status``/``scc_converged``
        make :meth:`ArrayBatchResult.failed_indices` unavailable with a
        precise error (keep those diagnostics on the host when you need them).
        """
        self._context._create()
        return _compute_array_batch(
            self,
            max_scc_iterations=max_scc_iterations,
            charge_tolerance=charge_tolerance,
            energy_tolerance=energy_tolerance,
            electronic_temperature=electronic_temperature,
            compute_energy=compute_energy,
            compute_forces=compute_forces,
            compute_charges=compute_charges,
            compute_point_charge_forces=compute_point_charge_forces,
            out=out,
            result_memory=result_memory,
        )

    def close(self) -> None:
        """Release this batch's native context."""
        self._context.close()

    def __enter__(self) -> ArrayBatch:  # noqa: PYI034 - Python 3.10 lacks typing.Self
        """Return this batch for use in a ``with`` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release native resources when leaving a ``with`` block."""
        self.close()


# --- packed Array API / DLPack execution helpers ------------------------------


def _require_all_or_none(
    group_name: str, fields: Sequence[str], namespace: dict[str, object]
) -> None:
    """Require every field of an optional descriptor group or none of them."""
    provided = [field for field in fields if namespace.get(field) is not None]
    if provided and len(provided) != len(fields):
        missing = [field for field in fields if field not in provided]
        raise XTBloomValueError(
            f"{group_name} descriptors must be supplied together; missing "
            + ", ".join(missing)
        )


def _array_shape(array: object) -> tuple[int, ...]:
    """Return the validated logical shape of one eager Array API array."""
    return _probe_shape(array)


def _compute_array_batch(
    batch: ArrayBatch,
    *,
    max_scc_iterations: int,
    charge_tolerance: float,
    energy_tolerance: float,
    electronic_temperature: float,
    compute_energy: bool,
    compute_forces: bool,
    compute_charges: bool,
    compute_point_charge_forces: bool | None,
    out: object | None,
    result_memory: str = "host",
) -> ArrayBatchResult:
    """Consume all DLPack views and run one synchronous compute call.

    The full request is validated before any output buffer is touched; every
    committed DLPack view is released exactly once on both the success and the
    failure path.
    """
    if result_memory not in ("host", "cuda"):
        raise XTBloomValueError(
            f"result_memory must be 'host' or 'cuda', got {result_memory!r}"
        )
    if result_memory == "cuda" and int(batch.backend) != library.BACKEND_CUDA:
        raise XTBloomNotSupportedError(
            "result_memory='cuda' requires the resolved CUDA backend; "
            "run a CUDA ArrayBatch or use result_memory='host'"
        )
    arrays = batch._arrays
    context = batch._context
    context._create()

    counts = _derive_batch_counts(arrays)
    nsystems, natoms, npoints, response_elements = counts

    out_spec = _normalize_out_spec(out)

    views: list[_dlpack.DLPackView] = []
    keepalive: list[object] = []
    output_owners: dict[str, object] = {}
    arenas: list[_dlpack._ResultArena] = []
    xtbloom_owned: list[_dlpack.DLPackResultBuffer] = []
    committed_result: ArrayBatchResult | None = None
    try:
        batch_descriptor = library.Batch()
        library._check_init(
            "xtbloom_batch_init",
            library.load_library().xtbloom_batch_init(
                ctypes.byref(batch_descriptor), ctypes.sizeof(batch_descriptor)
            ),
        )
        batch_descriptor.batch_size = nsystems
        batch_descriptor.total_atoms = natoms
        batch_descriptor.total_point_charges = npoints
        batch_descriptor.total_charge_response_elements = response_elements

        def consume_input(name: str, shape: tuple[int, ...]) -> _dlpack.DLPackView:
            array = arrays.get(name)
            if array is None:
                values = np.empty(0, dtype=_dlpack.EXPECTED_INPUT_DTYPES[name])
                keepalive.append(values)
                array = values
                shape = (0,)
            view = _dlpack.consume_from_dlpack(
                array,
                expected_dtype=_dlpack.EXPECTED_INPUT_DTYPES[name],
                expected_shape=shape,
                stream=context.stream,
                copy=batch._copy,
            )
            views.append(view)
            setattr(batch_descriptor, name, view.descriptor)
            return view

        # --- ragged batch inputs ------------------------------------------------
        consume_input("molecular_charges", (nsystems,))
        consume_input("atom_offsets", (nsystems + 1,))
        consume_input("atomic_numbers", (natoms,))
        consume_input("positions", (natoms, 3))
        consume_input("unpaired_electrons", (nsystems,))
        if arrays.get("spin_channels") is None:
            default_spin = np.full(nsystems, 1, dtype=np.int32)
            keepalive.append(default_spin)
            arrays["spin_channels"] = default_spin
        consume_input("spin_channels", (nsystems,))
        if arrays.get("point_charge_offsets") is not None:
            consume_input("point_charge_offsets", (nsystems + 1,))
            consume_input("point_charge_positions", (npoints, 3))
            consume_input("point_charge_values", (npoints,))
            consume_input("point_charge_gammas", (npoints,))
        if arrays.get("charge_response_matrix") is not None:
            consume_input("atomic_potential_shifts", (natoms,))
            consume_input("charge_response_offsets", (nsystems + 1,))
            consume_input("charge_response_matrix", (response_elements,))

        _validate_device_consistency(views, context)
        options = _build_compute_options(
            nsystems,
            npoints,
            max_scc_iterations=max_scc_iterations,
            charge_tolerance=charge_tolerance,
            energy_tolerance=energy_tolerance,
            electronic_temperature=electronic_temperature,
            compute_energy=compute_energy,
            compute_forces=compute_forces,
            compute_charges=compute_charges,
            compute_point_charge_forces=compute_point_charge_forces,
        )
        _validate_requested_outputs(out_spec, options.flags, npoints)
        result = library.BatchResult()
        library._check_init(
            "xtbloom_batch_result_init",
            library.load_library().xtbloom_batch_result_init(
                ctypes.byref(result), ctypes.sizeof(result)
            ),
        )
        _bind_outputs(
            result,
            out_spec,
            views,
            keepalive,
            output_owners,
            context.stream,
            nsystems,
            natoms,
            npoints,
            flags=options.flags,
            result_memory=result_memory,
            context=context,
            arenas=arenas,
            xtbloom_owned=xtbloom_owned,
        )
        # Output views are part of the same device contract as inputs. Check
        # them after binding, before native validation or any publication.
        _validate_device_consistency(views, context)
        library.compute_checked(context._create(), batch_descriptor, options, result)
        array_result = ArrayBatchResult(output_owners, result_flags=int(result.flags))
        array_result._attach_producers(arenas, xtbloom_owned)
        committed_result = array_result
        return array_result
    finally:
        _dlpack.release_all(views)
        if committed_result is None:
            # Uncommitted failure: no caller sees these handles, so every
            # xTBloom-owned arena/producer reference must be released now to
            # keep failure paths leak-free and allocation-counted.
            for producer in xtbloom_owned:
                producer.close()
            for arena in arenas:
                arena.close()


def _derive_batch_counts(arrays: dict[str, object | None]) -> tuple[int, int, int, int]:
    """Derive the ragged-batch extents from the input array shapes.

    Only metadata is read; native offset-value validation remains the
    authoritative check because it handles host and device storage
    identically.
    """
    molecular_shape = _array_shape(arrays["molecular_charges"])
    if len(molecular_shape) != 1:
        raise XTBloomValueError(
            f"molecular_charges must be one-dimensional, got {molecular_shape}"
        )
    nsystems = molecular_shape[0]
    if nsystems < 1:
        raise XTBloomValueError("a batch needs at least one system")
    atom_offsets_shape = _array_shape(arrays["atom_offsets"])
    if atom_offsets_shape != (nsystems + 1,):
        raise XTBloomValueError(
            f"atom_offsets must have shape ({nsystems + 1},), got {atom_offsets_shape}"
        )
    atomic_numbers_shape = _array_shape(arrays["atomic_numbers"])
    if len(atomic_numbers_shape) != 1:
        raise XTBloomValueError(
            f"atomic_numbers must be one-dimensional, got {atomic_numbers_shape}"
        )
    natoms = atomic_numbers_shape[0]
    if _array_shape(arrays["positions"]) != (natoms, 3):
        raise XTBloomValueError(
            f"positions must have shape ({natoms}, 3), got "
            f"{_array_shape(arrays['positions'])}"
        )
    npoints = 0
    if arrays["point_charge_offsets"] is not None:
        offsets_shape = _array_shape(arrays["point_charge_offsets"])
        if offsets_shape != (nsystems + 1,):
            raise XTBloomValueError(
                f"point_charge_offsets must have shape ({nsystems + 1},), "
                f"got {offsets_shape}"
            )
        point_positions_shape = _array_shape(arrays["point_charge_positions"])
        if len(point_positions_shape) != 2 or point_positions_shape[1] != 3:
            raise XTBloomValueError(
                "point_charge_positions must have shape (npoints, 3), got "
                f"{point_positions_shape}"
            )
        npoints = point_positions_shape[0]
        if _array_shape(arrays["point_charge_values"]) != (npoints,):
            raise XTBloomValueError("point charge values must match the position count")
        if _array_shape(arrays["point_charge_gammas"]) != (npoints,):
            raise XTBloomValueError("point charge gammas must match the position count")
    response_elements = 0
    if arrays["charge_response_matrix"] is not None:
        response_matrix_shape = _array_shape(arrays["charge_response_matrix"])
        if len(response_matrix_shape) != 1:
            raise XTBloomValueError(
                "charge_response_matrix must be one-dimensional, got "
                f"{response_matrix_shape}"
            )
        response_elements = response_matrix_shape[0]
        if _array_shape(arrays["charge_response_offsets"]) != (nsystems + 1,):
            raise XTBloomValueError(
                f"charge_response_offsets must have shape ({nsystems + 1},)"
            )
        if _array_shape(arrays["atomic_potential_shifts"]) != (natoms,):
            raise XTBloomValueError(
                f"atomic_potential_shifts must have shape ({natoms},)"
            )
    return nsystems, natoms, npoints, response_elements


def _validate_device_consistency(
    views: Sequence[_dlpack.DLPackView], context: Context
) -> None:
    """Require all CUDA views to belong to the context's resolved device.

    Host views are usable on any backend (a CUDA context stages host
    buffers).  CUDA views require the CUDA backend and the exact context
    device: xTBloom never performs an implicit cross-device copy.
    """
    cuda_views = [v for v in views if v.memory_space == library.MEMORY_CUDA_DEVICE]
    if not cuda_views:
        return
    if int(context.backend) != library.BACKEND_CUDA:
        raise XTBloomNotSupportedError("CUDA device arrays require the CUDA backend")
    resolved = int(context.device_id)
    for view in cuda_views:
        if not _same_device(view.device_id, resolved):
            raise XTBloomNotSupportedError(
                f"CUDA array on device {view.device_id} does not match the "
                f"context's resolved device {resolved}"
            )


def _same_device(device_id: int, resolved_device: int) -> bool:
    """Return whether one reported CUDA ordinal matches the context device."""
    return int(device_id) == int(resolved_device)


def _build_compute_options(
    nsystems: int,
    npoints: int,
    *,
    max_scc_iterations: int,
    charge_tolerance: float,
    energy_tolerance: float,
    electronic_temperature: float,
    compute_energy: bool,
    compute_forces: bool,
    compute_charges: bool,
    compute_point_charge_forces: bool | None,
) -> library.ComputeOptions:
    """Validate settings and build the ``xtbloom_compute_options_t`` mirror."""
    options = library.ComputeOptions()
    library._check_init(
        "xtbloom_compute_options_init",
        library.load_library().xtbloom_compute_options_init(
            ctypes.byref(options), ctypes.sizeof(options)
        ),
    )
    options.scc_start_mode = library.SCC_START_FRESH
    options.model = library.MODEL_GFN2_XTB
    flags = 0
    if compute_energy:
        flags |= library.COMPUTE_ENERGY
    if compute_forces:
        flags |= library.COMPUTE_FORCES
    if compute_charges:
        flags |= library.COMPUTE_ATOMIC_CHARGES
    if npoints and compute_point_charge_forces is not False:
        flags |= library.COMPUTE_POINT_CHARGE_FORCES
    options.flags = flags
    options.max_scc_iterations = _as_integer("max_scc_iterations", max_scc_iterations)
    options.charge_tolerance = float(charge_tolerance)
    options.energy_tolerance = float(energy_tolerance)
    options.electronic_temperature = (
        float(electronic_temperature) * library.KELVIN_TO_HARTREE
    )
    return options


def _normalize_out_spec(out: object | None) -> dict[str, object]:
    """Validate and normalize the ``out=`` output-policy mapping."""
    if out is None:
        return {}
    if not isinstance(out, dict):
        raise XTBloomValueError("out must be a mapping of output names to arrays")
    aliases = {"atomic_charges": "charges"}
    normalized: dict[str, object] = {}
    for name, array in out.items():
        canonical = aliases.get(name, name)
        if canonical not in _dlpack.EXPECTED_OUTPUT_DTYPES:
            raise XTBloomValueError(f"unknown output name {name!r}")
        if canonical in normalized:
            raise XTBloomValueError(f"output {name!r} was supplied more than once")
        if _array_adapter.is_lazy(array):
            raise XTBloomNotSupportedError(
                f"lazy/tracer objects cannot be used as {name} output buffers"
            )
        if _array_adapter.is_writable(array) is False:
            raise BufferError(
                f"output array {name!r} from {_array_adapter.backend_name(array)} "
                "is not writable; xTBloom never mutates read-only or JAX arrays"
            )
        normalized[canonical] = array
    return normalized


def _validate_requested_outputs(
    out_spec: dict[str, object], flags: int, npoints: int
) -> None:
    """Reject output buffers that the selected compute policy would ignore."""
    requested = {
        "energies": bool(flags & library.COMPUTE_ENERGY),
        "forces": bool(flags & library.COMPUTE_FORCES),
        "charges": bool(flags & library.COMPUTE_ATOMIC_CHARGES),
        "point_charge_forces": bool(
            npoints and flags & library.COMPUTE_POINT_CHARGE_FORCES
        ),
        # Native diagnostics are mandatory for every nonempty batch.
        "scc_iterations": True,
        "scc_converged": True,
        "per_system_status": True,
    }
    ignored = [name for name in out_spec if not requested[name]]
    if ignored:
        raise XTBloomValueError(
            "out contains buffers for properties that were not requested: "
            + ", ".join(ignored)
        )


def _bind_outputs(
    result: library.BatchResult,
    out_spec: dict[str, object],
    views: list[_dlpack.DLPackView],
    keepalive: list[object],
    output_owners: dict[str, object],
    stream: int | None,
    nsystems: int,
    natoms: int,
    npoints: int,
    *,
    flags: int,
    result_memory: str = "host",
    context: Context | None = None,
    arenas: list[_dlpack._ResultArena] | None = None,
    xtbloom_owned: list[_dlpack.DLPackResultBuffer] | None = None,
) -> None:
    """Bind every requested output buffer, honoring the ``out=`` policy.

    Omitted outputs follow ``result_memory``: ``"host"`` allocates a NumPy
    array as before, while ``"cuda"`` packs all xTBloom-owned slices into one
    device arena on the context device and hands back native
    :class:`DLPackResultBuffer` producers.  A supplied ``out=`` buffer always
    wins over the arena.
    """
    specs = (
        ("energies", "energies", (nsystems,), np.float64),
        ("forces", "forces", (natoms, 3), np.float64),
        ("charges", "atomic_charges", (natoms,), np.float64),
        ("point_charge_forces", "point_charge_forces", (npoints, 3), np.float64),
        ("scc_iterations", "scc_iterations", (nsystems,), np.int32),
        ("scc_converged", "scc_converged", (nsystems,), np.uint8),
        ("per_system_status", "per_system_status", (nsystems,), np.int32),
    )
    requested = _requested_output_mask(flags, npoints)
    if result_memory == "cuda":
        # A device arena can only be allocated on a live context; the
        # ``result_memory="cuda"`` callers always pass one.
        assert context is not None
        cuda_owned: list[tuple[str, str, tuple[int, ...], npt.DTypeLike]] = [
            (public_name, field_name, shape, dtype)
            for public_name, field_name, shape, dtype in specs
            if requested[public_name]
            and public_name not in out_spec
            and (shape and _dlpack._dlpack_bytes(shape, 1) != 0)
        ]
        if cuda_owned:
            arena, offsets = _allocate_result_arena(context, cuda_owned)
            arenas is not None and arenas.append(arena)
            base_pointer = arena.base_pointer()
            for public_name, field_name, shape, dtype in cuda_owned:
                dtype = np.dtype(dtype)
                view = _dlpack.DLPackResultBuffer(
                    arena=arena,
                    byte_offset=offsets[public_name],
                    size_bytes=_dlpack._dlpack_bytes(shape, dtype.itemsize),
                    shape=shape,
                    dtype=dtype,
                    memory_space=library.MEMORY_CUDA_DEVICE,
                    device_id=context.device_id,
                    stream=stream,
                )
                xtbloom_owned is not None and xtbloom_owned.append(view)
                setattr(
                    result,
                    field_name,
                    library.Buffer(
                        ctypes.c_void_p(base_pointer + offsets[public_name]),
                        view.size_bytes,
                        library.MEMORY_CUDA_DEVICE,
                        0,
                    ),
                )
                output_owners[public_name] = view
    for public_name, field_name, shape, dtype in specs:
        if not requested[public_name]:
            setattr(result, field_name, library.Buffer(None, 0, library.MEMORY_HOST, 0))
            continue
        if result_memory == "cuda" and public_name not in out_spec:
            continue
        owner = _bind_one_output(
            result,
            field_name,
            out_spec.get(public_name),
            shape,
            dtype,
            views,
            keepalive,
            stream,
        )
        if owner is not None:
            output_owners[public_name] = owner


def _requested_output_mask(flags: int, npoints: int) -> dict[str, bool]:
    """Return the requested-output mask for one compute flag set."""
    return {
        "energies": bool(flags & library.COMPUTE_ENERGY),
        "forces": bool(flags & library.COMPUTE_FORCES),
        "charges": bool(flags & library.COMPUTE_ATOMIC_CHARGES),
        "point_charge_forces": bool(
            npoints and flags & library.COMPUTE_POINT_CHARGE_FORCES
        ),
        "scc_iterations": True,
        "scc_converged": True,
        "per_system_status": True,
    }


def _allocate_result_arena(
    context: Context,
    outputs: list[tuple[str, str, tuple[int, ...], npt.DTypeLike]],
) -> tuple[_dlpack._ResultArena, dict[str, int]]:
    """Allocate one packed, alignment-checked device arena for ``outputs``.

    Slices are laid out at 64-byte granularity (the alignment DLPack
    consumers such as JAX, CuPy, and PyTorch expect for imported memory;
    ``cudaMalloc`` already returns 256-byte-aligned bases and the packed
    arena adds no per-slice alignment loss).  Returns the native arena wrapper
    and a mapping of public output name to byte offset.  All failures raise
    before any arena reference is leaked.
    """
    alignment = 64
    offset = 0
    offsets: dict[str, int] = {}
    for public_name, _, shape, dtype in outputs:
        dtype = np.dtype(dtype)
        byte_count = _dlpack._dlpack_bytes(shape, dtype.itemsize)
        if byte_count == 0:
            offsets[public_name] = 0
            continue
        if byte_count > _dlpack._POINTER_MAX:
            raise XTBloomRuntimeError(
                f"output {public_name} requires {byte_count} bytes, which "
                "exceeds the addressable arena extent"
            )
        offset = -(-offset // alignment) * alignment
        offsets[public_name] = offset
        offset += byte_count
    if offset == 0 or offset > _dlpack._POINTER_MAX:
        raise XTBloomValueError(
            "result_memory='cuda' requires at least one nonempty requested output "
            "within the addressable arena extent"
        )
    options = library.ResultOwnerOptions()
    library._check_init(
        "xtbloom_result_owner_options_init",
        library.load_library().xtbloom_result_owner_options_init(
            ctypes.byref(options), ctypes.sizeof(options)
        ),
    )
    options.memory_space = library.MEMORY_CUDA_DEVICE
    options.device_id = int(context.device_id)
    options.size_bytes = offset
    options.reserved = 0
    handle = ctypes.c_void_p()
    status = library.load_library().xtbloom_result_owner_create(
        ctypes.byref(options), ctypes.byref(handle)
    )
    if status != library.STATUS_SUCCESS:
        raise XTBloomRuntimeError(
            "xtbloom_result_owner_create failed with "
            f"{library.status_string(status)}: {library.get_last_error()}",
            status,
        )
    return _dlpack._ResultArena(handle), offsets


def _bind_one_output(
    result: library.BatchResult,
    field_name: str,
    array: object | None,
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    views: list[_dlpack.DLPackView],
    keepalive: list[object],
    stream: int | None,
) -> object | None:
    """Bind one output descriptor and return the array the result must expose.

    ``out=`` arrays are consumed as writable DLPack views; otherwise a host
    numpy owner is allocated and returned.  Empty outputs bind the null
    buffer and return ``None``.
    """
    resolved_dtype = np.dtype(dtype)
    if array is not None:
        actual = _array_shape(array)
        if actual != shape:
            raise XTBloomValueError(
                f"output array must have shape {shape}, got {actual}"
            )
        view = _dlpack.consume_from_dlpack(
            array,
            expected_dtype=resolved_dtype,
            expected_shape=shape,
            stream=stream,
            # Outputs must always alias the caller's buffer. Allowing the
            # batch input copy policy here would make a producer-side temporary
            # receive the result while the advertised ``out=`` array stays
            # unchanged.
            copy=False,
            writable_required=True,
            writable_hint=_array_adapter.is_writable(array),
        )
        views.append(view)
        setattr(result, field_name, view.as_output_buffer())
        return array
    if not shape:
        setattr(result, field_name, library.Buffer(None, 0, library.MEMORY_HOST, 0))
        return None
    owner = np.empty(shape, dtype=resolved_dtype)
    keepalive.append(owner)
    setattr(
        result,
        field_name,
        library.Buffer(
            ctypes.cast(owner.ctypes.data, ctypes.c_void_p),
            owner.nbytes,
            library.MEMORY_HOST,
            0,
        ),
    )
    return owner


class ArrayBatchResult:
    """Packed batch results whose arrays follow the caller's output policy.

    By default every array is a freshly allocated host numpy array.  When the
    associated ``out=`` buffers were supplied, the corresponding attributes
    reference those caller-owned arrays instead.  With
    ``result_memory="cuda"``, xTBloom-owned outputs are
    :class:`xtbloom._dlpack.DLPackResultBuffer` producers that importing
    frameworks consume through ``from_dlpack`` without a host copy.
    """

    def __init__(self, data: dict[str, object], result_flags: int) -> None:
        self._data = dict(data)
        self.result_flags = int(result_flags)
        self._arenas: list[_dlpack._ResultArena] = []
        self._producers: list[object] = []

    def _attach_producers(
        self,
        arenas: Sequence[_dlpack._ResultArena],
        producers: Sequence[object],
    ) -> None:
        """Keep xTBloom-owned arenas and DLPack producers alive with this result.

        Closing the result releases only the producer reference; each exported
        capsule and each live :class:`DLPackResultBuffer` independently retain
        the arena, so finished bytes survive this result and its compute
        context.
        """
        self._arenas.extend(arenas)
        self._producers.extend(producers)

    def _require(self, name: str) -> object:
        if name not in self._data:
            raise XTBloomValueError(f"{name} was not requested or has no storage")
        return self._data[name]

    @property
    def energies(self) -> object:
        """Per-system energies in Hartree, shape ``(nsystems,)``."""
        return self._require("energies")

    @property
    def forces(self) -> object:
        """Per-atom forces in Hartree/bohr, shape ``(natoms, 3)``."""
        return self._require("forces")

    @property
    def charges(self) -> object:
        """Per-atom partial charges, shape ``(natoms,)``."""
        return self._require("charges")

    @property
    def point_charge_forces(self) -> object:
        """Per-point-charge forces, or ``None`` when no points were present."""
        return self._data.get("point_charge_forces")

    @property
    def scc_iterations(self) -> object:
        """Per-system SCC iteration counts, shape ``(nsystems,)``."""
        return self._require("scc_iterations")

    @property
    def scc_converged(self) -> object:
        """Per-system SCC convergence flags, shape ``(nsystems,)``."""
        return self._require("scc_converged")

    @property
    def per_system_status(self) -> object:
        """Per-system native status codes, shape ``(nsystems,)``."""
        return self._require("per_system_status")

    @property
    def failed_indices(self) -> object:
        """Indices whose SCC/eigensolver status is not successful.

        Requires host numpy ``per_system_status``/``scc_converged`` (the
        default output policy).  Device-resident diagnostic arrays produced by
        ``result_memory="cuda"`` raise a precise error instead.
        """
        status = self._require("per_system_status")
        converged = self._require("scc_converged")
        if not isinstance(status, np.ndarray) or not isinstance(converged, np.ndarray):
            raise XTBloomNotSupportedError(
                "failed_indices requires host numpy status arrays; provide "
                "per_system_status/scc_converged without out= or keep them on the host"
            )
        return np.flatnonzero((status != library.STATUS_SUCCESS) | (converged != 1))

    def get(self, name: str) -> object:
        """Return one packed result array by name."""
        if name == "point_charge_forces":
            return self.point_charge_forces
        if name not in {
            "energies",
            "forces",
            "charges",
            "scc_iterations",
            "scc_converged",
            "per_system_status",
        }:
            raise XTBloomValueError(f"attribute {name!r} is not available")
        return self._require(name)

    def close(self) -> None:
        """Release the xTBloom-owned arena producer references (idempotent).

        Host NumPy and caller-owned ``out=`` arrays are unaffected. The
        returned :class:`DLPackResultBuffer` producers retain the arena
        independently, so they keep the finished bytes alive after this result
        is closed and can still be exported; their own ``close()``/``delete()``
        (or garbage collection) releases each of those references.
        """
        arenas, self._arenas = self._arenas, []
        for arena in arenas:
            arena.close()

    def __del__(self) -> None:
        """Release any xTBloom-owned arena if the result is not closed explicitly."""
        # Finalizers must never mask interpreter shutdown or user errors.
        with contextlib.suppress(Exception):
            self.close()

    def __enter__(self) -> ArrayBatchResult:  # noqa: PYI034 - 3.10 lacks Self
        """Use as a context manager so xTBloom-owned storage is released."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release xTBloom-owned storage when leaving a ``with`` block."""
        self.close()


def compute_arrays(
    atom_offsets: object,
    atomic_numbers: object,
    positions: object,
    molecular_charges: object,
    unpaired_electrons: object,
    spin_channels: object | None = None,
    point_charge_offsets: object | None = None,
    point_charge_positions: object | None = None,
    point_charge_values: object | None = None,
    point_charge_gammas: object | None = None,
    atomic_potential_shifts: object | None = None,
    charge_response_offsets: object | None = None,
    charge_response_matrix: object | None = None,
    *,
    copy: bool = False,
    backend: str | int = "auto",
    device_id: int | None = None,
    cpu_threads: int = 1,
    stream: int | None = None,
    max_scc_iterations: int = 250,
    charge_tolerance: float = 1.0e-6,
    energy_tolerance: float = 1.0e-8,
    electronic_temperature: float = 300.0,
    out: object | None = None,
    result_memory: str = "host",
) -> ArrayBatchResult:
    """One-shot packed inference; a convenience alias of :class:`ArrayBatch`.

    Builds a temporary :class:`ArrayBatch` from the flat descriptor arrays,
    computes with the given options, and returns an :class:`ArrayBatchResult`.
    ``out=`` and ``result_memory`` follow :meth:`ArrayBatch.compute`.
    """
    batch = ArrayBatch(
        atom_offsets,
        atomic_numbers,
        positions,
        molecular_charges,
        unpaired_electrons,
        spin_channels=spin_channels,
        point_charge_offsets=point_charge_offsets,
        point_charge_positions=point_charge_positions,
        point_charge_values=point_charge_values,
        point_charge_gammas=point_charge_gammas,
        atomic_potential_shifts=atomic_potential_shifts,
        charge_response_offsets=charge_response_offsets,
        charge_response_matrix=charge_response_matrix,
        copy=copy,
        backend=backend,
        device_id=device_id,
        cpu_threads=cpu_threads,
        stream=stream,
    )
    with batch:
        return batch.compute(
            max_scc_iterations=max_scc_iterations,
            charge_tolerance=charge_tolerance,
            energy_tolerance=energy_tolerance,
            electronic_temperature=electronic_temperature,
            out=out,
            result_memory=result_memory,
        )


__all__ = [
    "ELEMENT_SYMBOLS",
    "SYMBOL_TO_NUMBER",
    "ArrayBatch",
    "ArrayBatchResult",
    "BatchCalculator",
    "BatchResult",
    "Calculator",
    "ChargeResponse",
    "Context",
    "PointCharge",
    "Result",
    "Structure",
    "compute_arrays",
    "numbers_to_symbols",
    "symbols_to_numbers",
]
