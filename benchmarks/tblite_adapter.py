"""Persistent ctypes adapter for the tblite public C API."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class _StorageSlice(Protocol):
    """Atom range required from a public benchmark batch slice."""

    atom_begin: int
    atom_end: int


class _BatchStorage(Protocol):
    """Structural storage contract consumed by the tblite adapter."""

    atomic_numbers: Sequence[int]
    positions: Sequence[float]
    molecular_charges: Sequence[float]
    unpaired_electrons: Sequence[int]
    point_charge_values: Sequence[float]
    slices: Sequence[_StorageSlice]


_THREAD_RUNTIME_KEEPALIVE: list[ctypes.CDLL] = []


class TbliteError(RuntimeError):
    """A failing tblite API operation with its native diagnostic."""


def _configure_library(path: Path) -> ctypes.CDLL:
    """Load libtblite and declare every C symbol used by the benchmark."""
    if not path.is_file():
        raise TbliteError(f"tblite shared library is missing: {path}")
    try:
        library = ctypes.CDLL(str(path.resolve()))
    except OSError as exc:
        raise TbliteError(f"cannot load tblite shared library {path}: {exc}") from exc

    void_pointer = ctypes.c_void_p
    void_pointer_pointer = ctypes.POINTER(void_pointer)
    integer_pointer = ctypes.POINTER(ctypes.c_int)
    double_pointer = ctypes.POINTER(ctypes.c_double)

    library.tblite_get_version.argtypes = []
    library.tblite_get_version.restype = ctypes.c_int

    library.tblite_new_error.argtypes = []
    library.tblite_new_error.restype = void_pointer
    library.tblite_delete_error.argtypes = [void_pointer_pointer]
    library.tblite_delete_error.restype = None
    library.tblite_check_error.argtypes = [void_pointer]
    library.tblite_check_error.restype = ctypes.c_int
    library.tblite_get_error.argtypes = [
        void_pointer,
        ctypes.c_char_p,
        integer_pointer,
    ]
    library.tblite_get_error.restype = None

    library.tblite_new_context.argtypes = []
    library.tblite_new_context.restype = void_pointer
    library.tblite_delete_context.argtypes = [void_pointer_pointer]
    library.tblite_delete_context.restype = None
    library.tblite_check_context.argtypes = [void_pointer]
    library.tblite_check_context.restype = ctypes.c_int
    library.tblite_get_context_error.argtypes = [
        void_pointer,
        ctypes.c_char_p,
        integer_pointer,
    ]
    library.tblite_get_context_error.restype = None
    library.tblite_set_context_verbosity.argtypes = [void_pointer, ctypes.c_int]
    library.tblite_set_context_verbosity.restype = None

    library.tblite_new_structure.argtypes = [
        void_pointer,
        ctypes.c_int,
        integer_pointer,
        double_pointer,
        double_pointer,
        integer_pointer,
        double_pointer,
        ctypes.POINTER(ctypes.c_bool),
    ]
    library.tblite_new_structure.restype = void_pointer
    library.tblite_delete_structure.argtypes = [void_pointer_pointer]
    library.tblite_delete_structure.restype = None
    library.tblite_update_structure_geometry.argtypes = [
        void_pointer,
        void_pointer,
        double_pointer,
        double_pointer,
    ]
    library.tblite_update_structure_geometry.restype = None

    library.tblite_new_gfn2_calculator.argtypes = [
        void_pointer,
        void_pointer,
        void_pointer,
    ]
    library.tblite_new_gfn2_calculator.restype = void_pointer
    library.tblite_delete_calculator.argtypes = [void_pointer_pointer]
    library.tblite_delete_calculator.restype = None
    library.tblite_set_calculator_accuracy.argtypes = [
        void_pointer,
        void_pointer,
        ctypes.c_double,
    ]
    library.tblite_set_calculator_accuracy.restype = None
    library.tblite_set_calculator_max_iter.argtypes = [
        void_pointer,
        void_pointer,
        ctypes.c_int,
    ]
    library.tblite_set_calculator_max_iter.restype = None
    library.tblite_set_calculator_temperature.argtypes = [
        void_pointer,
        void_pointer,
        ctypes.c_double,
    ]
    library.tblite_set_calculator_temperature.restype = None

    library.tblite_new_result.argtypes = []
    library.tblite_new_result.restype = void_pointer
    library.tblite_delete_result.argtypes = [void_pointer_pointer]
    library.tblite_delete_result.restype = None
    library.tblite_get_singlepoint.argtypes = [
        void_pointer,
        void_pointer,
        void_pointer,
        void_pointer,
    ]
    library.tblite_get_singlepoint.restype = None
    library.tblite_get_result_energy.argtypes = [
        void_pointer,
        void_pointer,
        double_pointer,
    ]
    library.tblite_get_result_energy.restype = None
    library.tblite_get_result_gradient.argtypes = [
        void_pointer,
        void_pointer,
        double_pointer,
    ]
    library.tblite_get_result_gradient.restype = None
    library.tblite_get_result_charges.argtypes = [
        void_pointer,
        void_pointer,
        double_pointer,
    ]
    library.tblite_get_result_charges.restype = None
    return library


def _load_first(candidates: list[str | None]) -> ctypes.CDLL | None:
    """Load the first usable thread runtime and retain it process-wide."""
    for candidate in dict.fromkeys(value for value in candidates if value):
        try:
            runtime = ctypes.CDLL(candidate)
        except OSError:
            continue
        _THREAD_RUNTIME_KEEPALIVE.append(runtime)
        return runtime
    return None


def _enforce_single_thread(library_directory: Path) -> dict[str, Any]:
    """Pin the OpenMP and BLAS runtimes before tblite numerical work."""
    return _configure_runtime_threads(library_directory, 1)


def _configure_runtime_threads(library_directory: Path, threads: int) -> dict[str, Any]:
    """Expose the requested OpenMP/OpenBLAS thread count to the reference engine.

    The cross-engine benchmark runs every engine with the same worker budget so
    the three figures compare equal resources. ``threads=1`` reproduces the
    original single-thread-pinned rows; ``threads>1`` lets OpenMP-enabled
    reference builds use their worker threads exactly like gpuxtb's worker
    pool and dxtb's Torch intra-op threads.
    """
    thread_text = str(int(threads))
    os.environ["OMP_NUM_THREADS"] = thread_text
    os.environ["OPENBLAS_NUM_THREADS"] = thread_text
    os.environ["MKL_NUM_THREADS"] = thread_text
    controls: dict[str, Any] = {
        "OMP_NUM_THREADS": thread_text,
        "OPENBLAS_NUM_THREADS": thread_text,
        "MKL_NUM_THREADS": thread_text,
        "omp_set_num_threads": False,
        "openblas_set_num_threads": False,
    }
    openmp = _load_first(
        [
            str(library_directory / "libgomp.so.1"),
            ctypes.util.find_library("gomp"),
            "libgomp.so.1",
        ]
    )
    if openmp is not None:
        try:
            openmp.omp_set_dynamic.argtypes = [ctypes.c_int]
            openmp.omp_set_dynamic.restype = None
            openmp.omp_set_num_threads.argtypes = [ctypes.c_int]
            openmp.omp_set_num_threads.restype = None
            openmp.omp_set_dynamic(0)
            openmp.omp_set_num_threads(threads)
            controls["omp_set_num_threads"] = True
        except AttributeError:
            pass
    blas = _load_first(
        [
            str(library_directory / "libopenblas.so.0"),
            ctypes.util.find_library("openblas"),
            "libopenblas.so.0",
        ]
    )
    if blas is not None:
        try:
            blas.openblas_set_num_threads.argtypes = [ctypes.c_int]
            blas.openblas_set_num_threads.restype = None
            blas.openblas_set_num_threads(threads)
            controls["openblas_set_num_threads"] = True
        except AttributeError:
            pass
    return controls


def _delete(library: ctypes.CDLL, function_name: str, handle: ctypes.c_void_p) -> None:
    """Delete one opaque tblite handle through its pointer-to-handle ABI."""
    if not handle:
        return
    owned = ctypes.c_void_p(handle.value)
    getattr(library, function_name)(ctypes.byref(owned))


@dataclass
class TbliteState:
    """One persistent tblite system and caller-owned result storage."""

    error: ctypes.c_void_p
    context: ctypes.c_void_p
    structure: ctypes.c_void_p
    calculator: ctypes.c_void_p
    result: ctypes.c_void_p
    positions: Any
    energy: ctypes.c_double
    gradient: Any | None
    charges: Any | None
    keepalive: tuple[Any, ...]


class TbliteAdapter:
    """Execute a homogeneous logical batch as a serial tblite C API loop."""

    supports_external_point_charges = False
    external_point_charge_reason = (
        "tblite public C API exports uniform electric-field containers but no "
        "discrete external point-charge potential or point-charge-force getter"
    )

    def __init__(
        self,
        library_path: Path,
        storage: _BatchStorage,
        property_name: str,
        accuracy: float = 1.0e-4,
        max_iterations: int = 500,
        electronic_temperature_hartree: float = 300.0 * 3.166808578545117e-6,
        collect_atomic_charges: bool = True,
        threads: int = 1,
    ) -> None:
        if storage.point_charge_values:
            raise TbliteError(self.external_point_charge_reason)
        if type(threads) is not int or threads <= 0:
            raise TbliteError("tblite threads must be a positive integer")
        self.library_path = library_path
        self.threads = threads
        self.thread_control = _configure_runtime_threads(
            library_path.resolve().parent, threads
        )
        self.library = _configure_library(library_path)
        self.version = int(self.library.tblite_get_version())
        self.storage = storage
        self.property_name = property_name
        self.accuracy = accuracy
        self.max_iterations = max_iterations
        self.electronic_temperature_hartree = electronic_temperature_hartree
        self.collect_atomic_charges = collect_atomic_charges
        self.states: list[TbliteState] = []
        try:
            for index in range(len(storage.slices)):
                self.states.append(self._create_state(index))
        except BaseException:
            self.close()
            raise

    def _error_message(self, error: ctypes.c_void_p) -> str:
        buffer = ctypes.create_string_buffer(8192)
        size = ctypes.c_int(ctypes.sizeof(buffer))
        self.library.tblite_get_error(error, buffer, ctypes.byref(size))
        return buffer.value.decode("utf-8", errors="replace").strip()

    def _check_error(self, error: ctypes.c_void_p, operation: str) -> None:
        if self.library.tblite_check_error(error) == 0:
            return
        message = self._error_message(error)
        raise TbliteError(f"{operation} failed: {message or 'empty tblite error'}")

    def _check_context(self, context: ctypes.c_void_p, operation: str) -> None:
        if self.library.tblite_check_context(context) == 0:
            return
        messages: list[str] = []
        for _ in range(32):
            if self.library.tblite_check_context(context) == 0:
                break
            buffer = ctypes.create_string_buffer(8192)
            size = ctypes.c_int(ctypes.sizeof(buffer))
            self.library.tblite_get_context_error(context, buffer, ctypes.byref(size))
            message = buffer.value.decode("utf-8", errors="replace").strip()
            messages.append(message or "empty tblite context error")
        raise TbliteError(f"{operation} failed: {'; '.join(messages)}")

    def _configure_calculator(
        self, context: ctypes.c_void_p, calculator: ctypes.c_void_p
    ) -> None:
        """Apply the benchmark's SCC settings to one calculator."""
        self.library.tblite_set_calculator_accuracy(context, calculator, self.accuracy)
        self.library.tblite_set_calculator_max_iter(
            context, calculator, self.max_iterations
        )
        self.library.tblite_set_calculator_temperature(
            context, calculator, self.electronic_temperature_hartree
        )
        self._check_context(context, "configure tblite calculator")

    def restart_scc(self) -> None:
        """Drop the converged density and restart every system from SAD.

        tblite 0.7 exposes no public wavefunction-reset call, so a genuinely
        cold measured sample rebuilds the calculator (and result) while keeping
        the persistent context, structure, and caller-owned buffers. This makes
        a ``--cold-samples`` panel-1 row the real cold-start comparison instead
        of the warm continuation a persistent adapter would otherwise get.
        """
        for state in self.states:
            old_result = state.result
            old_calculator = state.calculator
            state.result = ctypes.c_void_p()
            state.calculator = ctypes.c_void_p()
            _delete(self.library, "tblite_delete_result", old_result)
            _delete(self.library, "tblite_delete_calculator", old_calculator)

            calculator = ctypes.c_void_p()
            result = ctypes.c_void_p()
            try:
                calculator = ctypes.c_void_p(
                    self.library.tblite_new_gfn2_calculator(
                        state.context, state.structure, None
                    )
                )
                self._check_context(state.context, "tblite_new_gfn2_calculator")
                if not calculator:
                    raise TbliteError("tblite calculator allocation returned NULL")
                self._configure_calculator(state.context, calculator)
                result = ctypes.c_void_p(self.library.tblite_new_result())
                if not result:
                    raise TbliteError("tblite result allocation returned NULL")
            except BaseException:
                _delete(self.library, "tblite_delete_result", result)
                _delete(self.library, "tblite_delete_calculator", calculator)
                raise
            state.calculator = calculator
            state.result = result

    def _create_state(self, index: int) -> TbliteState:
        """Create one complete persistent state outside measured calls."""
        item = self.storage.slices[index]
        atom_count = item.atom_end - item.atom_begin
        atomic_numbers = (ctypes.c_int * atom_count)(
            *self.storage.atomic_numbers[item.atom_begin : item.atom_end]
        )
        positions = (ctypes.c_double * (3 * atom_count))(
            *self.storage.positions[3 * item.atom_begin : 3 * item.atom_end]
        )
        charge = ctypes.c_double(self.storage.molecular_charges[index])
        unpaired = ctypes.c_int(self.storage.unpaired_electrons[index])
        error = ctypes.c_void_p(self.library.tblite_new_error())
        context = ctypes.c_void_p(self.library.tblite_new_context())
        structure = ctypes.c_void_p()
        calculator = ctypes.c_void_p()
        result = ctypes.c_void_p()
        try:
            if not error or not context:
                raise TbliteError("tblite error/context allocation returned NULL")
            self.library.tblite_set_context_verbosity(context, 0)
            self._check_context(context, "tblite_set_context_verbosity")
            structure = ctypes.c_void_p(
                self.library.tblite_new_structure(
                    error,
                    atom_count,
                    atomic_numbers,
                    positions,
                    ctypes.byref(charge),
                    ctypes.byref(unpaired),
                    None,
                    None,
                )
            )
            self._check_error(error, "tblite_new_structure")
            if not structure:
                raise TbliteError("tblite_new_structure returned NULL")
            calculator = ctypes.c_void_p(
                self.library.tblite_new_gfn2_calculator(context, structure, None)
            )
            self._check_context(context, "tblite_new_gfn2_calculator")
            self._configure_calculator(context, calculator)
            result = ctypes.c_void_p(self.library.tblite_new_result())
            if not calculator or not result:
                raise TbliteError("tblite calculator/result allocation returned NULL")
            gradient = (
                (ctypes.c_double * (3 * atom_count))()
                if self.property_name == "force"
                else None
            )
            charges = (
                (ctypes.c_double * atom_count)()
                if self.collect_atomic_charges
                else None
            )
            return TbliteState(
                error=error,
                context=context,
                structure=structure,
                calculator=calculator,
                result=result,
                positions=positions,
                energy=ctypes.c_double(),
                gradient=gradient,
                charges=charges,
                keepalive=(atomic_numbers, charge, unpaired),
            )
        except BaseException:
            _delete(self.library, "tblite_delete_result", result)
            _delete(self.library, "tblite_delete_calculator", calculator)
            _delete(self.library, "tblite_delete_structure", structure)
            _delete(self.library, "tblite_delete_context", context)
            _delete(self.library, "tblite_delete_error", error)
            raise

    def invoke(self) -> None:
        """Run the serial logical batch, including geometry update and getters."""
        for state in self.states:
            self.library.tblite_update_structure_geometry(
                state.error, state.structure, state.positions, None
            )
            self._check_error(state.error, "tblite_update_structure_geometry")
            self.library.tblite_get_singlepoint(
                state.context, state.structure, state.calculator, state.result
            )
            self._check_context(state.context, "tblite_get_singlepoint")
            self.library.tblite_get_result_energy(
                state.error, state.result, ctypes.byref(state.energy)
            )
            self._check_error(state.error, "tblite_get_result_energy")
            if state.gradient is not None:
                self.library.tblite_get_result_gradient(
                    state.error, state.result, state.gradient
                )
                self._check_error(state.error, "tblite_get_result_gradient")
            if state.charges is not None:
                self.library.tblite_get_result_charges(
                    state.error, state.result, state.charges
                )
                self._check_error(state.error, "tblite_get_result_charges")

    def results(self) -> dict[str, Any]:
        """Normalize tblite gradients to forces and retain atomic charges."""
        output: dict[str, Any] = {
            "energies_hartree": [float(state.energy.value) for state in self.states]
        }
        if getattr(self, "collect_atomic_charges", True):
            output["atomic_charges_e"] = [
                float(value) for state in self.states for value in (state.charges or ())
            ]
        if self.property_name == "force":
            output["forces_hartree_per_bohr"] = [
                -float(value)
                for state in self.states
                for value in (state.gradient or ())
            ]
        return output

    def close(self) -> None:
        """Release persistent states in reverse construction order."""
        while self.states:
            state = self.states.pop()
            _delete(self.library, "tblite_delete_result", state.result)
            _delete(self.library, "tblite_delete_calculator", state.calculator)
            _delete(self.library, "tblite_delete_structure", state.structure)
            _delete(self.library, "tblite_delete_context", state.context)
            _delete(self.library, "tblite_delete_error", state.error)
