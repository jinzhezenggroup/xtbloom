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
* ``spin_channels`` selects restricted (1) or unrestricted (2) orbitals. On the
  CPU backend the default is unrestricted for open-shell systems and restricted
  otherwise. The current CUDA descriptor boundary only supports restricted,
  closed-shell systems.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

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
    *["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb"],
    *["Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"],
    *["Tl", "Pb", "Bi", "Po", "At", "Rn"],
    *["Fr", "Ra"],
    *["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No"],
    *["Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"],
    *["Nh", "Fl", "Mc", "Lv", "Ts", "Og"],
]

SYMBOL_TO_NUMBER = {symbol: number + 1 for number, symbol in enumerate(ELEMENT_SYMBOLS)}


def symbols_to_numbers(symbols: Sequence[str]) -> List[int]:
    """Convert a list of atomic symbols to atomic numbers."""
    return [SYMBOL_TO_NUMBER[symbol] for symbol in symbols]


def numbers_to_symbols(numbers: Sequence[int]) -> List[str]:
    """Convert a list of atomic numbers to atomic symbols."""
    return [ELEMENT_SYMBOLS[int(number) - 1] for number in numbers]


# --- supported methods -------------------------------------------------------------

_SUPPORTED_METHODS = {
    "GFN2-xTB": library.MODEL_GFN2_XTB,
    "GFN2": library.MODEL_GFN2_XTB,
}

_BACKENDS = {"auto": library.BACKEND_AUTO, "cpu": library.BACKEND_CPU, "cuda": library.BACKEND_CUDA}

CUDA_UNRESTRICTED_SCOPE_REASON = (
    "the CUDA public GFN2 path does not support ABI-v2 unrestricted spin_channels yet"
)
CUDA_OPEN_SHELL_SCOPE_REASON = (
    "the CUDA public GFN2 path does not support open-shell systems "
    "(nonzero unpaired electrons) yet"
)


def _default_spin_channels(uhf: int) -> int:
    """Pick the tblite-compatible spin-polarization default for a spin state."""
    return 2 if uhf != 0 else 1


def _resolve_uhf(uhf: Optional[int], multiplicity: Optional[int]) -> int:
    """Resolve the number of unpaired electrons from ``uhf`` and/or ``multiplicity``."""
    if multiplicity is not None:
        multiplicity = int(multiplicity)
        if multiplicity < 1:
            raise GPUxtbValueError("multiplicity must be a positive integer")
        unpaired = multiplicity - 1
        if uhf is not None and int(uhf) != unpaired:
            raise GPUxtbValueError(
                f"uhf={uhf} is inconsistent with multiplicity={multiplicity}"
            )
        return unpaired
    return int(uhf) if uhf is not None else 0


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
        object.__setattr__(self, "positions", np.ascontiguousarray(positions))
        object.__setattr__(self, "charges", np.ascontiguousarray(charges))
        object.__setattr__(self, "gammas", np.ascontiguousarray(gammas))


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
    ):
        numbers = np.asarray(numbers)
        if numbers.ndim == 1 and numbers.dtype.kind in "USO":
            numbers = np.asarray(symbols_to_numbers(numbers), dtype=np.int64)
        else:
            numbers = np.asarray(numbers, dtype=np.int64)
        positions = np.asarray(positions, dtype=float)

        if numbers.ndim != 1 or numbers.size == 0:
            raise GPUxtbValueError("expected a nonempty one-dimensional numbers array")
        if (numbers < 1).any() or (numbers > 118).any():
            raise GPUxtbValueError("atomic numbers must lie between 1 and 118")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise GPUxtbValueError("positions must have shape (n, 3)")
        if positions.shape[0] != numbers.size:
            raise GPUxtbValueError("dimension mismatch between numbers and positions")
        if not np.isfinite(positions).all():
            raise GPUxtbValueError("positions must be finite")

        unpaired = _resolve_uhf(uhf, multiplicity)
        if spin_channels is not None and spin_channels not in (1, 2):
            raise GPUxtbValueError("spin_channels must be 1 (restricted) or 2 (unrestricted)")

        self._numbers = np.ascontiguousarray(numbers, dtype=np.int32)
        self._positions = np.ascontiguousarray(positions, dtype=np.float64)
        self._charge = float(charge)
        self._uhf = unpaired
        self._spin_channels = spin_channels if spin_channels is not None else _default_spin_channels(unpaired)
        self._point_charges = point_charges

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
        if positions is not None:
            positions = np.asarray(positions, dtype=float)
            if positions.shape != self._positions.shape:
                raise GPUxtbValueError("updated positions must keep the original shape")
            if not np.isfinite(positions).all():
                raise GPUxtbValueError("positions must be finite")
            self._positions = np.ascontiguousarray(positions, dtype=np.float64)
        if charge is not None:
            self._charge = float(charge)
        if uhf is not None or multiplicity is not None:
            self._uhf = _resolve_uhf(uhf, multiplicity)
            if spin_channels is None:
                self._spin_channels = _default_spin_channels(self._uhf)
        if spin_channels is not None:
            if spin_channels not in (1, 2):
                raise GPUxtbValueError("spin_channels must be 1 (restricted) or 2 (unrestricted)")
            self._spin_channels = spin_channels


# --- contexts ----------------------------------------------------------------------


class Context:
    """Ownership wrapper around a ``gpuxtb_context_t``.

    The C context keeps its worker pools and numerical caches for the lifetime
    of the object, so a single context should be reused across geometry
    updates and steady-state calls. Creates and destroys the native context on
    enter/exit.
    """

    _handle: Optional[ctypes.c_void_p] = None

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
            if int(backend) not in (library.BACKEND_AUTO, library.BACKEND_CPU, library.BACKEND_CUDA):
                raise GPUxtbValueError(f"unknown backend {backend!r}")
            self._requested = int(backend)
        self._device_id = -1 if device_id is None else int(device_id)
        self._cpu_threads = int(cpu_threads)
        if self._cpu_threads < 0:
            raise GPUxtbValueError("cpu_threads must be nonnegative")
        self._handle = None
        self._backend: Optional[int] = None

    def _create(self) -> None:
        if self._handle is not None:
            return
        options = library.ContextOptions()
        library._check_init(
            "gpuxtb_context_options_init",
            library.load_library().gpuxtb_context_options_init(
                ctypes.byref(options), ctypes.sizeof(options)
            ),
        )
        options.backend = self._requested
        options.device_id = self._device_id
        options.cpu_threads = self._cpu_threads
        handle = ctypes.c_void_p()
        status = library.load_library().gpuxtb_context_create(
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
        self._backend = library.load_library().gpuxtb_context_get_backend(handle)

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
            library.load_library().gpuxtb_context_destroy(self._handle)
            self._handle = None
            self._backend = None

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


def _validate_for_backend(structures: Sequence[Structure], backend: int) -> None:
    """Enforce the current per-backend descriptor limits early and clearly."""
    if backend == library.BACKEND_CUDA:
        for index, structure in enumerate(structures):
            if structure.uhf != 0:
                raise GPUxtbNotSupportedError(
                    f"structure {index}: {CUDA_OPEN_SHELL_SCOPE_REASON}"
                )
            if structure.spin_channels != 1:
                raise GPUxtbNotSupportedError(
                    f"structure {index}: {CUDA_UNRESTRICTED_SCOPE_REASON}"
                )


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
    backend = context.backend
    _validate_for_backend(structures, backend)

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
    batch.total_charge_response_elements = 0

    def bind(descriptor_name, values, ctype, dtype):
        if not values:
            setattr(batch, descriptor_name, library.ConstBuffer(None, 0, library.MEMORY_HOST, 0))
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
    # The CUDA descriptor boundary rejects any active spin-channel buffer.
    if backend != library.BACKEND_CUDA:
        bind("spin_channels", spin_channels, ctypes.c_int32, np.int32)
    if total_points:
        bind("point_charge_offsets", point_offsets, ctypes.c_int64, np.int64)
        bind("point_charge_positions", point_positions, ctypes.c_double, np.float64)
        bind("point_charge_values", point_values, ctypes.c_double, np.float64)
        bind("point_charge_gammas", point_gammas, ctypes.c_double, np.float64)

    # --- compute options -------------------------------------------------------
    options = library.ComputeOptions()
    library._check_init(
        "gpuxtb_compute_options_init",
        library_instance.gpuxtb_compute_options_init(
            ctypes.byref(options), ctypes.sizeof(options)
        ),
    )
    options.model = model
    options.flags = flags
    options.max_scc_iterations = int(max_scc_iterations)
    options.charge_tolerance = float(charge_tolerance)
    options.energy_tolerance = float(energy_tolerance)
    options.electronic_temperature = float(electronic_temperature) * library.KELVIN_TO_HARTREE

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
    point_charge_forces = np.empty((total_points, 3), dtype=np.float64) if total_points else None
    scc_iterations = np.empty(nsystems, dtype=np.int32)
    scc_converged = np.empty(nsystems, dtype=np.uint8)
    per_system_status = np.empty(nsystems, dtype=np.int32)

    def bind_output(buffer_field, owner, requested):
        if not requested or owner is None:
            setattr(result, buffer_field, library.Buffer(None, 0, library.MEMORY_HOST, 0))
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
        point_offsets=np.asarray(point_offsets, dtype=np.int64) if total_points else None,
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
                f"scc_converged={int(converged)}, iterations={int(computed.scc_iterations[index])}"
            )
    if failed:
        raise GPUxtbRuntimeError("gpuxtb batch inference produced failed systems: " + "; ".join(failed))


# --- results ----------------------------------------------------------------------


class Result:
    """Single-system results container, similar to ``tblite.interface.Result``."""

    _getter = {
        "energy": lambda self: self.energy,
        "energies": lambda self: np.asarray([self.energy]),
        "forces": lambda self: self.forces,
        "gradient": lambda self: self.forces,
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

        Available keys: ``energy``, ``energies``, ``forces`` (alias
        ``gradient``), ``charges``, ``scc_iterations``, ``scc_converged``,
        ``scc_status``, and ``natoms``.
        """
        if attribute not in self._getter:
            raise GPUxtbValueError(f"attribute {attribute!r} is not available in this result")
        return self._getter[attribute](self)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def dict(self) -> dict:
        return {key: self.get(key) for key in self._getter}


class BatchResult:
    """Multi-system results container for :class:`BatchCalculator`."""

    def __init__(self, computed: _ComputedBatch, structures: Sequence[Structure]) -> None:
        nsystems = len(structures)
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
            raise GPUxtbValueError(f"attribute {attribute!r} is not available in this result")
        return names[attribute]


# --- calculators -------------------------------------------------------------------


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
        self.max_scc_iterations = max_scc_iterations
        self.charge_tolerance = charge_tolerance
        self.energy_tolerance = energy_tolerance
        self.electronic_temperature = electronic_temperature


def _resolve_method(method: str) -> int:
    if not method:
        raise GPUxtbValueError("a method must be provided (only GFN2-xTB is currently supported)")
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
        *,
        backend: Union[str, int] = "auto",
        device_id: Optional[int] = None,
        cpu_threads: int = 1,
        max_scc_iterations: int = 250,
        charge_tolerance: float = 1.0e-6,
        energy_tolerance: float = 1.0e-8,
        electronic_temperature: float = 300.0,
    ):
        if max_scc_iterations < 1:
            raise GPUxtbValueError("max_scc_iterations must be positive")
        if charge_tolerance <= 0.0:
            raise GPUxtbValueError("charge_tolerance must be positive")
        if energy_tolerance <= 0.0:
            raise GPUxtbValueError("energy_tolerance must be positive")
        if electronic_temperature < 0.0:
            raise GPUxtbValueError("electronic_temperature must be nonnegative")

        Structure.__init__(
            self,
            numbers,
            positions,
            charge,
            uhf=uhf,
            multiplicity=multiplicity,
            spin_channels=spin_channels,
            point_charges=point_charges,
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
        if attribute == "max_scc_iterations":
            self._settings.max_scc_iterations = int(value)
        elif attribute == "charge_tolerance":
            self._settings.charge_tolerance = float(value)
        elif attribute == "energy_tolerance":
            self._settings.energy_tolerance = float(value)
        elif attribute == "electronic_temperature":
            self._settings.electronic_temperature = float(value)
        else:
            raise GPUxtbValueError(f"unsupported calculator setting {attribute!r}")

    def singlepoint(self) -> Result:
        """Perform a single-point calculation and return a :class:`Result`."""
        flags = library.COMPUTE_ENERGY | library.COMPUTE_FORCES | library.COMPUTE_ATOMIC_CHARGES
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
            int(max_scc_iterations),
            float(charge_tolerance),
            float(energy_tolerance),
            float(electronic_temperature),
        )
        self._context = Context(backend, device_id, cpu_threads)

    def __len__(self) -> int:
        return len(self._structures)

    @property
    def backend(self) -> int:
        return self._context.backend

    def set(self, attribute: str, value: Any) -> None:
        if attribute == "max_scc_iterations":
            self._settings.max_scc_iterations = int(value)
        elif attribute == "charge_tolerance":
            self._settings.charge_tolerance = float(value)
        elif attribute == "energy_tolerance":
            self._settings.energy_tolerance = float(value)
        elif attribute == "electronic_temperature":
            self._settings.electronic_temperature = float(value)
        else:
            raise GPUxtbValueError(f"unsupported calculator setting {attribute!r}")

    def compute(self) -> BatchResult:
        """Run the whole batch and return a :class:`BatchResult`."""
        flags = library.COMPUTE_ENERGY | library.COMPUTE_FORCES | library.COMPUTE_ATOMIC_CHARGES
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
        _raise_on_failure(computed)
        return BatchResult(computed, self._structures)

    def close(self) -> None:
        self._context.close()


__all__ = [
    "ELEMENT_SYMBOLS",
    "SYMBOL_TO_NUMBER",
    "symbols_to_numbers",
    "numbers_to_symbols",
    "PointCharge",
    "Structure",
    "Context",
    "Result",
    "BatchResult",
    "Calculator",
    "BatchCalculator",
]