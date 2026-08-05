"""High-level Python interface to the gpuxtb public C ABI.

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

import ctypes
import math
import operator
import weakref
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from . import library
from .exceptions import GPUxtbNotSupportedError, GPUxtbRuntimeError, GPUxtbValueError

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


def _as_integer(name: str, value: Any) -> int:
    """Return an exact integer without silently truncating floats."""
    if isinstance(value, (bool, np.bool_)):
        raise GPUxtbValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError:
        raise GPUxtbValueError(f"{name} must be an integer") from None


def symbols_to_numbers(symbols: Sequence[str]) -> List[int]:
    """Convert a list of atomic symbols to atomic numbers."""
    return [SYMBOL_TO_NUMBER[symbol] for symbol in symbols]


def numbers_to_symbols(numbers: Sequence[int]) -> List[str]:
    """Convert a list of atomic numbers to atomic symbols."""
    symbols = []
    for value in numbers:
        number = _as_integer("atomic number", value)
        if number < 1 or number > len(ELEMENT_SYMBOLS):
            raise GPUxtbValueError("atomic numbers must lie between 1 and 118")
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


def _resolve_uhf(uhf: Optional[int], multiplicity: Optional[int]) -> int:
    """Resolve the number of unpaired electrons from ``uhf`` and/or ``multiplicity``."""
    if multiplicity is not None:
        multiplicity = _as_integer("multiplicity", multiplicity)
        if multiplicity < 1:
            raise GPUxtbValueError("multiplicity must be a positive integer")
        unpaired = multiplicity - 1
        if uhf is not None and _as_integer("uhf", uhf) != unpaired:
            raise GPUxtbValueError(
                f"uhf={uhf} is inconsistent with multiplicity={multiplicity}"
            )
        return unpaired
    unpaired = _as_integer("uhf", uhf) if uhf is not None else 0
    if unpaired < 0:
        raise GPUxtbValueError("uhf must be nonnegative")
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
        positions = np.asarray(self.positions, dtype=float)
        charges = np.asarray(self.charges, dtype=float)
        gammas = np.asarray(self.gammas, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise GPUxtbValueError("point charge positions must have shape (n, 3)")
        if charges.shape != (len(positions),):
            raise GPUxtbValueError("point charge values must match the position count")
        if gammas.shape != (len(positions),):
            raise GPUxtbValueError("point charge gammas must match the position count")
        if len(positions) == 0:
            raise GPUxtbValueError("point charges must be nonempty")
        if not (
            np.isfinite(positions).all()
            and np.isfinite(charges).all()
            and np.isfinite(gammas).all()
        ):
            raise GPUxtbValueError("point charge inputs must be finite")
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
    coordinates are outside gpuxtb and are not included in the forces.

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
        shifts = np.asarray(self.shifts, dtype=float)
        matrix = np.asarray(self.matrix, dtype=float)
        if shifts.ndim != 1:
            raise GPUxtbValueError("charge response shifts must be one-dimensional")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise GPUxtbValueError("charge response matrix must be square")
        if matrix.shape[0] != shifts.size:
            raise GPUxtbValueError("charge response matrix must match the shift count")
        if not (np.isfinite(shifts).all() and np.isfinite(matrix).all()):
            raise GPUxtbValueError("charge response inputs must be finite")
        object.__setattr__(self, "shifts", np.ascontiguousarray(shifts))
        object.__setattr__(self, "matrix", np.ascontiguousarray(matrix))


# --- structures --------------------------------------------------------------------


class Structure:
    """A molecular structure with an immutable atom count and atomic species.

    Coordinates (bohr), total charge, number of unpaired electrons, and spin
    channels can be updated through :meth:`update` without rebuilding the
    object.
    """

    def __init__(
        self,
        numbers: Union[np.ndarray, List[int], Sequence[str]],
        positions: np.ndarray,
        charge: float = 0.0,
        uhf: Optional[int] = None,
        multiplicity: Optional[int] = None,
        spin_channels: Optional[int] = None,
        point_charges: Optional[PointCharge] = None,
        charge_response: Optional[ChargeResponse] = None,
    ):
        object_numbers = np.asarray(numbers, dtype=object)
        raw_numbers = np.asarray(numbers)
        if raw_numbers.ndim != 1 or raw_numbers.size == 0:
            raise GPUxtbValueError("expected a nonempty one-dimensional numbers array")

        object_symbols = raw_numbers.dtype.kind == "O" and all(
            isinstance(value, str) for value in raw_numbers
        )
        if raw_numbers.dtype.kind in "US" or object_symbols:
            numbers = np.asarray(symbols_to_numbers(raw_numbers), dtype=np.int64)
        else:
            if raw_numbers.dtype.kind in "bc" or (
                any(
                    isinstance(value, (bool, np.bool_)) for value in object_numbers.flat
                )
            ):
                raise GPUxtbValueError("atomic numbers must be exact integers")
            try:
                numeric_numbers = np.asarray(raw_numbers, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                raise GPUxtbValueError(
                    "atomic numbers must be exact integers"
                ) from None
            if (
                not np.isfinite(numeric_numbers).all()
                or not np.equal(numeric_numbers, np.floor(numeric_numbers)).all()
            ):
                raise GPUxtbValueError("atomic numbers must be exact integers")
            if (numeric_numbers < 1).any() or (numeric_numbers > 118).any():
                raise GPUxtbValueError("atomic numbers must lie between 1 and 118")
            numbers = np.asarray(numeric_numbers, dtype=np.int64)
        positions = np.asarray(positions, dtype=float)

        if (numbers < 1).any() or (numbers > 118).any():
            raise GPUxtbValueError("atomic numbers must lie between 1 and 118")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise GPUxtbValueError("positions must have shape (n, 3)")
        if positions.shape[0] != numbers.size:
            raise GPUxtbValueError("dimension mismatch between numbers and positions")
        if not np.isfinite(positions).all():
            raise GPUxtbValueError("positions must be finite")

        unpaired = _resolve_uhf(uhf, multiplicity)
        if not math.isfinite(float(charge)):
            raise GPUxtbValueError("charge must be finite")
        resolved_spin = (
            _default_spin_channels(unpaired)
            if spin_channels is None
            else _as_integer("spin_channels", spin_channels)
        )
        if resolved_spin not in (1, 2):
            raise GPUxtbValueError(
                "spin_channels must be 1 (restricted) or 2 (unrestricted)"
            )

        self._numbers = np.ascontiguousarray(numbers, dtype=np.int32)
        self._positions = np.ascontiguousarray(positions, dtype=np.float64)
        self._charge = float(charge)
        self._uhf = unpaired
        self._spin_channels = resolved_spin
        self._point_charges = point_charges
        self._charge_response = charge_response

    def __len__(self) -> int:
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
    def point_charges(self) -> Optional[PointCharge]:
        """External point charges attached to this structure, if any."""
        return self._point_charges

    @property
    def charge_response(self) -> Optional[ChargeResponse]:
        """Periodic SCC shift ``b`` and response matrix ``A``, if any."""
        return self._charge_response

    def update(
        self,
        positions: Optional[np.ndarray] = None,
        charge: Optional[float] = None,
        uhf: Optional[int] = None,
        multiplicity: Optional[int] = None,
        spin_channels: Optional[int] = None,
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
                raise GPUxtbValueError("updated positions must keep the original shape")
            if not np.isfinite(candidate_positions).all():
                raise GPUxtbValueError("positions must be finite")
            next_positions = np.ascontiguousarray(candidate_positions, dtype=np.float64)
        if charge is not None:
            if not math.isfinite(float(charge)):
                raise GPUxtbValueError("charge must be finite")
            next_charge = float(charge)
        if uhf is not None or multiplicity is not None:
            next_uhf = _resolve_uhf(uhf, multiplicity)
            if spin_channels is None:
                next_spin_channels = _default_spin_channels(next_uhf)
        if spin_channels is not None:
            next_spin_channels = _as_integer("spin_channels", spin_channels)
            if next_spin_channels not in (1, 2):
                raise GPUxtbValueError(
                    "spin_channels must be 1 (restricted) or 2 (unrestricted)"
                )

        self._positions = next_positions
        self._charge = next_charge
        self._uhf = next_uhf
        self._spin_channels = next_spin_channels


# --- contexts ----------------------------------------------------------------------


def _destroy_native_context(library_instance, handle: ctypes.c_void_p) -> None:
    """Destroy one native context; used by explicit close and finalization."""
    library_instance.gpuxtb_context_destroy(handle)


class Context:
    """Ownership wrapper around a ``gpuxtb_context_t``.

    The C context keeps its worker pools and numerical caches for the lifetime
    of the object, so a single context should be reused across geometry
    updates and steady-state calls. Creates and destroys the native context on
    enter/exit.
    """

    def __init__(
        self,
        backend: Union[str, int] = "auto",
        device_id: Optional[int] = None,
        cpu_threads: int = 1,
    ) -> None:
        if isinstance(backend, str):
            try:
                self._requested = _BACKENDS[backend]
            except KeyError:
                raise GPUxtbValueError(f"unknown backend {backend!r}") from None
        else:
            if int(backend) not in (
                library.BACKEND_AUTO,
                library.BACKEND_CPU,
                library.BACKEND_CUDA,
            ):
                raise GPUxtbValueError(f"unknown backend {backend!r}")
            self._requested = int(backend)
        self._device_id = -1 if device_id is None else int(device_id)
        self._cpu_threads = int(cpu_threads)
        if self._cpu_threads < 0:
            raise GPUxtbValueError("cpu_threads must be nonnegative")
        self._handle = None
        self._backend: Optional[int] = None
        self._finalizer: Optional[weakref.finalize] = None

    def _create(self) -> None:
        if self._handle is not None:
            return
        library_instance = library.load_library()
        options = library.ContextOptions()
        library._check_init(
            "gpuxtb_context_options_init",
            library_instance.gpuxtb_context_options_init(
                ctypes.byref(options), ctypes.sizeof(options)
            ),
        )
        options.backend = self._requested
        options.device_id = self._device_id
        options.cpu_threads = self._cpu_threads
        handle = ctypes.c_void_p()
        status = library_instance.gpuxtb_context_create(
            ctypes.byref(options), ctypes.byref(handle)
        )
        if status == library.STATUS_BACKEND_UNAVAILABLE:
            raise GPUxtbRuntimeError(
                f"backend unavailable: {library.get_last_error()}", status
            )
        if status != library.STATUS_SUCCESS:
            raise GPUxtbRuntimeError(
                f"gpuxtb_context_create failed with {library.status_string(status)}: "
                f"{library.get_last_error()}",
                status,
            )
        self._handle = handle
        self._backend = library_instance.gpuxtb_context_get_backend(handle)
        # Keep both the CDLL and the native handle alive in the finalizer so
        # contexts are reclaimed even when users follow the concise examples
        # and do not call close() explicitly.
        self._finalizer = weakref.finalize(
            self, _destroy_native_context, library_instance, handle
        )

    @property
    def backend(self) -> int:
        """The resolved backend (CPU or CUDA)."""
        self._create()
        return int(self._backend)

    @property
    def device_id(self) -> int:
        """The backend device id (``-1`` for CPU)."""
        self._create()
        return int(library.load_library().gpuxtb_context_get_device_id(self._handle))

    def close(self) -> None:
        if self._handle is not None:
            if self._finalizer is not None and self._finalizer.alive:
                self._finalizer()
            self._handle = None
            self._backend = None
            self._finalizer = None

    def __enter__(self) -> "Context":
        self._create()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


# --- low-level batch execution ----------------------------------------------------


@dataclass
class _ComputedBatch:
    """Raw native results and diagnostics for one ragged batch."""

    energies: np.ndarray
    forces: np.ndarray
    charges: np.ndarray
    point_charge_forces: Optional[np.ndarray]
    scc_iterations: np.ndarray
    scc_converged: np.ndarray
    per_system_status: np.ndarray
    result_flags: int
    atom_offsets: np.ndarray
    point_offsets: Optional[np.ndarray]
    keepalive: list


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
    for structure, response in zip(structures, responses):
        atom_count = len(structure)
        if response is None:
            shifts.extend(0.0 for _ in range(atom_count))
            matrices.extend(0.0 for _ in range(atom_count * atom_count))
        else:
            if response.shifts.size != atom_count:
                raise GPUxtbValueError(
                    "charge response shifts must match the atom count"
                )
            shifts.extend(float(value) for value in response.shifts)
            matrices.extend(float(value) for value in response.matrix.ravel())
        offsets.append(len(matrices))
    return offsets, shifts, matrices


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
) -> _ComputedBatch:
    """Populate descriptors and run one synchronous ``gpuxtb_compute`` call."""
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
        atom_offsets.append(len(atomic_numbers))
        point_offsets.append(len(point_values))

    total_atoms = len(atomic_numbers)
    total_points = len(point_values)

    # --- bind descriptors ------------------------------------------------------
    library_instance = library.load_library()
    batch = library.Batch()
    library._check_init(
        "gpuxtb_batch_init",
        library_instance.gpuxtb_batch_init(ctypes.byref(batch), ctypes.sizeof(batch)),
    )
    batch.batch_size = len(structures)
    batch.total_atoms = total_atoms
    batch.total_point_charges = total_points
    batch.total_charge_response_elements = (
        len(packed_responses[2]) if packed_responses is not None else 0
    )

    def bind(descriptor_name, values, ctype, dtype):
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

    bind("atom_offsets", atom_offsets, ctypes.c_int64, np.int64)
    bind("atomic_numbers", atomic_numbers, ctypes.c_int32, np.int32)
    bind("positions", positions, ctypes.c_double, np.float64)
    bind("molecular_charges", molecular_charges, ctypes.c_double, np.float64)
    bind("unpaired_electrons", unpaired_electrons, ctypes.c_int32, np.int32)
    bind("spin_channels", spin_channels, ctypes.c_int32, np.int32)
    if total_points:
        bind("point_charge_offsets", point_offsets, ctypes.c_int64, np.int64)
        bind("point_charge_positions", point_positions, ctypes.c_double, np.float64)
        bind("point_charge_values", point_values, ctypes.c_double, np.float64)
        bind("point_charge_gammas", point_gammas, ctypes.c_double, np.float64)
    if packed_responses is not None:
        response_offsets, response_shifts, response_matrix = packed_responses
        bind("atomic_potential_shifts", response_shifts, ctypes.c_double, np.float64)
        bind("charge_response_offsets", response_offsets, ctypes.c_int64, np.int64)
        bind("charge_response_matrix", response_matrix, ctypes.c_double, np.float64)

    # --- compute options -------------------------------------------------------
    options = library.ComputeOptions()
    library._check_init(
        "gpuxtb_compute_options_init",
        library_instance.gpuxtb_compute_options_init(
            ctypes.byref(options), ctypes.sizeof(options)
        ),
    )
    # High-level calculators intentionally use reproducible independent SCC
    # solves; persistent warm-start policy remains a low-level ABI feature.
    options.scc_start_mode = library.SCC_START_FRESH
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
        "gpuxtb_batch_result_init",
        library_instance.gpuxtb_batch_result_init(
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
    scc_iterations = np.empty(nsystems, dtype=np.int32)
    scc_converged = np.empty(nsystems, dtype=np.uint8)
    per_system_status = np.empty(nsystems, dtype=np.int32)

    def bind_output(buffer_field, owner, requested):
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
    bind_output("scc_iterations", scc_iterations, True)
    bind_output("scc_converged", scc_converged, True)
    bind_output("per_system_status", per_system_status, True)

    library.compute_checked(context._handle, batch, options, result)

    return _ComputedBatch(
        energies=energies,
        forces=forces,
        charges=charges,
        point_charge_forces=point_charge_forces,
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
        zip(computed.per_system_status, computed.scc_converged)
    ):
        if int(status) != library.STATUS_SUCCESS or int(converged) != 1:
            failed.append(
                f"system {index}: {library.status_string(int(status))}, "
                f"scc_converged={int(converged)}, "
                f"iterations={int(computed.scc_iterations[index])}"
            )
    if failed:
        raise GPUxtbRuntimeError(
            "gpuxtb batch inference produced failed systems: " + "; ".join(failed)
        )


# --- results ----------------------------------------------------------------------


class Result:
    """Single-system results container, similar to ``tblite.interface.Result``."""

    _getter = {
        "energy": lambda self: self.energy,
        "energies": lambda self: np.asarray([self.energy]),
        "forces": lambda self: self.forces,
        # The public C ABI returns force = -dE/dR, whereas tblite-style
        # ``gradient`` means +dE/dR.
        "gradient": lambda self: -self.forces,
        "charges": lambda self: self.charges,
        "point_charge_forces": lambda self: self.point_charge_forces,
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
        self.scc_iterations = int(computed.scc_iterations[index])
        self.scc_converged = bool(computed.scc_converged[index])
        self.scc_status = int(computed.per_system_status[index])

    @property
    def natoms(self) -> int:
        return self._natoms

    def get(self, attribute: str) -> Any:
        """Return a requested quantity by name.

        Available keys: ``energy``, ``energies``, ``forces``, ``gradient``
        (the negative of forces), ``charges``, ``scc_iterations``,
        ``scc_converged``, ``scc_status``, and ``natoms``.
        """
        if attribute not in self._getter:
            raise GPUxtbValueError(
                f"attribute {attribute!r} is not available in this result"
            )
        return self._getter[attribute](self)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def dict(self) -> dict:
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
        return len(self._structures)

    def __getitem__(self, index: int) -> Result:
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
        failed = []
        for index in self.failed_indices:
            failed.append(
                f"system {int(index)}: "
                f"{library.status_string(int(self.per_system_status[index]))}, "
                f"scc_converged={int(self.scc_converged[index])}, "
                f"iterations={int(self.scc_iterations[index])}"
            )
        if failed:
            raise GPUxtbRuntimeError(
                "gpuxtb batch inference produced failed systems: " + "; ".join(failed)
            )

    def get(self, attribute: str) -> Any:
        """Return scalar batch arrays by name (``energies``, ``forces``, ...)."""
        names = {
            "energies": self.energies,
            "forces": self.forces,
            "charges": self.charges,
            "point_charge_forces": self.point_charge_forces,
            "scc_iterations": self.scc_iterations,
            "scc_converged": self.scc_converged,
            "per_system_status": self.per_system_status,
        }
        if attribute not in names:
            raise GPUxtbValueError(
                f"attribute {attribute!r} is not available in this result"
            )
        return names[attribute]


# --- calculators -------------------------------------------------------------------


def _validated_compute_setting(attribute: str, value: Any) -> Union[int, float]:
    """Validate one compute setting and return its normalized scalar value."""
    if attribute == "max_scc_iterations":
        candidate = _as_integer(attribute, value)
        if candidate <= 0:
            raise GPUxtbValueError("max_scc_iterations must be positive")
        return candidate
    if attribute in ("charge_tolerance", "energy_tolerance"):
        candidate = float(value)
        if not math.isfinite(candidate) or candidate <= 0.0:
            raise GPUxtbValueError(f"{attribute} must be finite and positive")
        return candidate
    if attribute == "electronic_temperature":
        candidate = float(value)
        if not math.isfinite(candidate) or candidate < 0.0:
            raise GPUxtbValueError(
                "electronic_temperature must be finite and nonnegative"
            )
        return candidate
    raise GPUxtbValueError(f"unsupported calculator setting {attribute!r}")


class _ComputeSettings:
    __slots__ = (
        "model",
        "max_scc_iterations",
        "charge_tolerance",
        "energy_tolerance",
        "electronic_temperature",
    )

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

    def set(self, attribute: str, value: Any) -> None:
        """Validate and transactionally update one public compute option."""
        setattr(self, attribute, _validated_compute_setting(attribute, value))


def _resolve_method(method: str) -> int:
    if not method:
        raise GPUxtbValueError(
            "a method must be provided (only GFN2-xTB is currently supported)"
        )
    try:
        return _SUPPORTED_METHODS[method]
    except KeyError:
        if method in ("GFN1-xTB", "GFN1"):
            raise GPUxtbNotSupportedError(
                "GFN1-xTB is reserved by the gpuxtb ABI but is not implemented yet"
            ) from None
        raise GPUxtbValueError(f"unknown method {method!r}") from None


class Calculator(Structure):
    """Single-point GFN2-xTB calculator for one structure (tblite-like API).

    Example
    -------
    >>> import numpy as np
    >>> from gpuxtb.interface import Calculator
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
    """

    def __init__(
        self,
        method: str,
        numbers: Union[np.ndarray, List[int], Sequence[str]],
        positions: np.ndarray,
        charge: float = 0.0,
        uhf: Optional[int] = None,
        multiplicity: Optional[int] = None,
        spin_channels: Optional[int] = None,
        point_charges: Optional[PointCharge] = None,
        charge_response: Optional[ChargeResponse] = None,
        *,
        backend: Union[str, int] = "auto",
        device_id: Optional[int] = None,
        cpu_threads: int = 1,
        max_scc_iterations: int = 250,
        charge_tolerance: float = 1.0e-6,
        energy_tolerance: float = 1.0e-8,
        electronic_temperature: float = 300.0,
    ):
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

    @property
    def backend(self) -> int:
        """The resolved execution backend of this calculator."""
        return self._context.backend

    @property
    def method(self) -> str:
        return self._method

    def update(
        self,
        positions: Optional[np.ndarray] = None,
        charge: Optional[float] = None,
        uhf: Optional[int] = None,
        multiplicity: Optional[int] = None,
        spin_channels: Optional[int] = None,
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

    def set(self, attribute: str, value: Any) -> None:
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
        computed = _compute_batch(
            self._context,
            [self],
            model=self._settings.model,
            max_scc_iterations=self._settings.max_scc_iterations,
            charge_tolerance=self._settings.charge_tolerance,
            energy_tolerance=self._settings.energy_tolerance,
            electronic_temperature=self._settings.electronic_temperature,
            flags=flags,
        )
        _raise_on_failure(computed)
        return Result(computed, index=0)

    def close(self) -> None:
        self._context.close()

    def __enter__(self) -> "Calculator":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class BatchCalculator:
    """Batched GFN2-xTB calculator over many structures in one C call.

    The C API describes a ragged batch with flat arrays and offsets, so all
    systems are solved together while keeping per-system convergence state.
    """

    def __init__(
        self,
        structures: Sequence[Structure],
        method: str = "GFN2-xTB",
        *,
        backend: Union[str, int] = "auto",
        device_id: Optional[int] = None,
        cpu_threads: int = 1,
        max_scc_iterations: int = 250,
        charge_tolerance: float = 1.0e-6,
        energy_tolerance: float = 1.0e-8,
        electronic_temperature: float = 300.0,
    ):
        if not structures:
            raise GPUxtbValueError("a batch needs at least one structure")
        self._structures = list(structures)
        self._settings = _ComputeSettings(
            _resolve_method(method),
            max_scc_iterations,
            charge_tolerance,
            energy_tolerance,
            electronic_temperature,
        )
        self._context = Context(backend, device_id, cpu_threads)

    def __len__(self) -> int:
        return len(self._structures)

    @property
    def backend(self) -> int:
        return self._context.backend

    def set(self, attribute: str, value: Any) -> None:
        self._settings.set(attribute, value)

    def compute(self, *, raise_on_failure: bool = False) -> BatchResult:
        """Run the batch while preserving successful peers.

        By default the result is returned even when individual systems fail;
        their floating-point slices contain NaNs and diagnostics identify the
        failed peers.  Set ``raise_on_failure=True`` or call
        :meth:`BatchResult.raise_for_status` for strict behavior.
        """
        flags = (
            library.COMPUTE_ENERGY
            | library.COMPUTE_FORCES
            | library.COMPUTE_ATOMIC_CHARGES
        )
        if any(structure.point_charges is not None for structure in self._structures):
            flags |= library.COMPUTE_POINT_CHARGE_FORCES
        computed = _compute_batch(
            self._context,
            self._structures,
            model=self._settings.model,
            max_scc_iterations=self._settings.max_scc_iterations,
            charge_tolerance=self._settings.charge_tolerance,
            energy_tolerance=self._settings.energy_tolerance,
            electronic_temperature=self._settings.electronic_temperature,
            flags=flags,
        )
        batch_result = BatchResult(computed, self._structures)
        if raise_on_failure:
            batch_result.raise_for_status()
        return batch_result

    def close(self) -> None:
        self._context.close()

    def __enter__(self) -> "BatchCalculator":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "ELEMENT_SYMBOLS",
    "SYMBOL_TO_NUMBER",
    "symbols_to_numbers",
    "numbers_to_symbols",
    "PointCharge",
    "ChargeResponse",
    "Structure",
    "Context",
    "Result",
    "BatchResult",
    "Calculator",
    "BatchCalculator",
]
