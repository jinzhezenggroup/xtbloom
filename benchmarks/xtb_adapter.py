"""Persistent ctypes adapter for the xTB 6.7.1 public C API."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

XTB_VERBOSITY_MUTED = 0
XTB_API_VERSION_1_0_0 = 10000
_THREAD_RUNTIME_KEEPALIVE: list[ctypes.CDLL] = []


class XtbError(RuntimeError):
    """A failing xTB C API operation with its environment error stack."""


def _configure_library(path: Path) -> ctypes.CDLL:
    """Load libxtb and declare every C symbol used by the benchmark."""
    if not path.is_file():
        raise XtbError(f"xTB shared library is missing: {path}")
    try:
        library = ctypes.CDLL(str(path.resolve()))
    except OSError as exc:
        raise XtbError(f"cannot load xTB shared library {path}: {exc}") from exc

    void_pointer = ctypes.c_void_p
    void_pointer_pointer = ctypes.POINTER(void_pointer)
    integer_pointer = ctypes.POINTER(ctypes.c_int)
    double_pointer = ctypes.POINTER(ctypes.c_double)

    library.xtb_getAPIVersion.argtypes = []
    library.xtb_getAPIVersion.restype = ctypes.c_int
    library.xtb_newEnvironment.argtypes = []
    library.xtb_newEnvironment.restype = void_pointer
    library.xtb_delEnvironment.argtypes = [void_pointer_pointer]
    library.xtb_delEnvironment.restype = None
    library.xtb_checkEnvironment.argtypes = [void_pointer]
    library.xtb_checkEnvironment.restype = ctypes.c_int
    library.xtb_getError.argtypes = [void_pointer, ctypes.c_char_p, integer_pointer]
    library.xtb_getError.restype = None
    library.xtb_setVerbosity.argtypes = [void_pointer, ctypes.c_int]
    library.xtb_setVerbosity.restype = None

    library.xtb_newMolecule.argtypes = [
        void_pointer,
        integer_pointer,
        integer_pointer,
        double_pointer,
        double_pointer,
        integer_pointer,
        double_pointer,
        ctypes.POINTER(ctypes.c_bool),
    ]
    library.xtb_newMolecule.restype = void_pointer
    library.xtb_delMolecule.argtypes = [void_pointer_pointer]
    library.xtb_delMolecule.restype = None
    library.xtb_updateMolecule.argtypes = [
        void_pointer,
        void_pointer,
        double_pointer,
        double_pointer,
    ]
    library.xtb_updateMolecule.restype = None

    library.xtb_newCalculator.argtypes = []
    library.xtb_newCalculator.restype = void_pointer
    library.xtb_delCalculator.argtypes = [void_pointer_pointer]
    library.xtb_delCalculator.restype = None
    library.xtb_loadGFN2xTB.argtypes = [
        void_pointer,
        void_pointer,
        void_pointer,
        ctypes.c_char_p,
    ]
    library.xtb_loadGFN2xTB.restype = None
    library.xtb_setAccuracy.argtypes = [void_pointer, void_pointer, ctypes.c_double]
    library.xtb_setAccuracy.restype = None
    library.xtb_setMaxIter.argtypes = [void_pointer, void_pointer, ctypes.c_int]
    library.xtb_setMaxIter.restype = None
    library.xtb_setElectronicTemp.argtypes = [
        void_pointer,
        void_pointer,
        ctypes.c_double,
    ]
    library.xtb_setElectronicTemp.restype = None
    library.xtb_setExternalCharges.argtypes = [
        void_pointer,
        void_pointer,
        integer_pointer,
        integer_pointer,
        double_pointer,
        double_pointer,
    ]
    library.xtb_setExternalCharges.restype = None
    library.xtb_releaseExternalCharges.argtypes = [void_pointer, void_pointer]
    library.xtb_releaseExternalCharges.restype = None

    library.xtb_newResults.argtypes = []
    library.xtb_newResults.restype = void_pointer
    library.xtb_delResults.argtypes = [void_pointer_pointer]
    library.xtb_delResults.restype = None
    library.xtb_singlepoint.argtypes = [
        void_pointer,
        void_pointer,
        void_pointer,
        void_pointer,
    ]
    library.xtb_singlepoint.restype = None
    library.xtb_getEnergy.argtypes = [void_pointer, void_pointer, double_pointer]
    library.xtb_getEnergy.restype = None
    library.xtb_getGradient.argtypes = [void_pointer, void_pointer, double_pointer]
    library.xtb_getGradient.restype = None
    library.xtb_getPCGradient.argtypes = [void_pointer, void_pointer, double_pointer]
    library.xtb_getPCGradient.restype = None
    return library


def _enforce_single_thread(library_directory: Path) -> dict[str, Any]:
    """Pin OpenMP/OpenBLAS before libxtb performs any numerical work."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    controls: dict[str, Any] = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "omp_set_num_threads": False,
        "openblas_set_num_threads": False,
    }
    try:
        openmp = ctypes.CDLL(str(library_directory / "libgomp.so.1"))
        openmp.omp_set_dynamic.argtypes = [ctypes.c_int]
        openmp.omp_set_dynamic.restype = None
        openmp.omp_set_num_threads.argtypes = [ctypes.c_int]
        openmp.omp_set_num_threads.restype = None
        openmp.omp_set_dynamic(0)
        openmp.omp_set_num_threads(1)
        _THREAD_RUNTIME_KEEPALIVE.append(openmp)
        controls["omp_set_num_threads"] = True
    except (OSError, AttributeError):
        pass
    try:
        blas = ctypes.CDLL(str(library_directory / "libopenblas.so.0"))
        blas.openblas_set_num_threads.argtypes = [ctypes.c_int]
        blas.openblas_set_num_threads.restype = None
        blas.openblas_set_num_threads(1)
        _THREAD_RUNTIME_KEEPALIVE.append(blas)
        controls["openblas_set_num_threads"] = True
    except (OSError, AttributeError):
        pass
    return controls


def _delete(library: ctypes.CDLL, function_name: str, handle: ctypes.c_void_p) -> None:
    """Delete one opaque xTB handle using its pointer-to-handle ABI."""
    if not handle:
        return
    function = getattr(library, function_name)
    owned = ctypes.c_void_p(handle.value)
    function(ctypes.byref(owned))


@dataclass
class XtbState:
    """One fully persistent xTB system and caller-owned output storage."""

    environment: ctypes.c_void_p
    molecule: ctypes.c_void_p
    calculator: ctypes.c_void_p
    result: ctypes.c_void_p
    positions: Any
    energy: ctypes.c_double
    gradient: Any | None
    point_gradient: Any | None
    has_external_charges: bool
    point_count: ctypes.c_int | None
    point_numbers: Any | None
    point_charges: Any | None
    point_positions: Any | None
    keepalive: tuple[Any, ...]


class XtbAdapter:
    """Execute a homogeneous logical batch as a serial loop over C API states."""

    def __init__(
        self,
        library_path: Path,
        storage: Any,
        property_name: str,
        point_source_atomic_numbers: list[int] | None,
        accuracy: float = 1.0e-4,
        max_iterations: int = 500,
        electronic_temperature_kelvin: float = 300.0,
    ) -> None:
        self.library_path = library_path
        self.thread_control = _enforce_single_thread(library_path.resolve().parent)
        self.library = _configure_library(library_path)
        self.api_version = int(self.library.xtb_getAPIVersion())
        if self.api_version < XTB_API_VERSION_1_0_0:
            raise XtbError(
                f"xTB C API {self.api_version} is older than required API 10000"
            )
        self.storage = storage
        self.property_name = property_name
        self.accuracy = accuracy
        self.max_iterations = max_iterations
        self.electronic_temperature_kelvin = electronic_temperature_kelvin
        self.point_source_atomic_numbers = point_source_atomic_numbers
        self.states: list[XtbState] = []
        try:
            for index in range(len(storage.slices)):
                self.states.append(self._create_state(index))
        except BaseException:
            self.close()
            raise

    def _check(self, environment: ctypes.c_void_p, operation: str) -> None:
        """Raise with xTB's error stack and clear it for deterministic recovery."""
        if self.library.xtb_checkEnvironment(environment) == 0:
            return
        buffer = ctypes.create_string_buffer(8192)
        size = ctypes.c_int(ctypes.sizeof(buffer))
        self.library.xtb_getError(environment, buffer, ctypes.byref(size))
        message = buffer.value.decode("utf-8", errors="replace").strip()
        raise XtbError(f"{operation} failed: {message or 'xTB error stack was empty'}")

    def _create_state(self, index: int) -> XtbState:
        """Create and configure one molecule/calculator outside measured calls."""
        item = self.storage.slices[index]
        atom_count = item.atom_end - item.atom_begin
        point_count = item.point_end - item.point_begin
        atomic_numbers = (ctypes.c_int * atom_count)(
            *self.storage.atomic_numbers[item.atom_begin : item.atom_end]
        )
        positions = (ctypes.c_double * (3 * atom_count))(
            *self.storage.positions[3 * item.atom_begin : 3 * item.atom_end]
        )
        charge = ctypes.c_double(self.storage.molecular_charges[index])
        unpaired = ctypes.c_int(self.storage.unpaired_electrons[index])
        natoms = ctypes.c_int(atom_count)
        environment = ctypes.c_void_p(self.library.xtb_newEnvironment())
        molecule = ctypes.c_void_p()
        calculator = ctypes.c_void_p()
        result = ctypes.c_void_p()
        try:
            if not environment:
                raise XtbError("xtb_newEnvironment returned NULL")
            self.library.xtb_setVerbosity(environment, XTB_VERBOSITY_MUTED)
            self._check(environment, "xtb_setVerbosity")
            molecule = ctypes.c_void_p(
                self.library.xtb_newMolecule(
                    environment,
                    ctypes.byref(natoms),
                    atomic_numbers,
                    positions,
                    ctypes.byref(charge),
                    ctypes.byref(unpaired),
                    None,
                    None,
                )
            )
            self._check(environment, "xtb_newMolecule")
            if not molecule:
                raise XtbError("xtb_newMolecule returned NULL")
            calculator = ctypes.c_void_p(self.library.xtb_newCalculator())
            result = ctypes.c_void_p(self.library.xtb_newResults())
            if not calculator or not result:
                raise XtbError("xTB calculator or result allocation returned NULL")
            self.library.xtb_loadGFN2xTB(environment, molecule, calculator, None)
            self._check(environment, "xtb_loadGFN2xTB")
            self.library.xtb_setAccuracy(environment, calculator, self.accuracy)
            self.library.xtb_setMaxIter(environment, calculator, self.max_iterations)
            self.library.xtb_setElectronicTemp(
                environment, calculator, self.electronic_temperature_kelvin
            )
            self._check(environment, "configure xTB calculator")
            if point_count:
                if (
                    self.point_source_atomic_numbers is None
                    or len(self.point_source_atomic_numbers) != point_count
                ):
                    raise XtbError(
                        "QM/MM xTB adapter requires one source atomic number per point"
                    )
                point_numbers = (ctypes.c_int * point_count)(
                    *self.point_source_atomic_numbers
                )
                point_charges = (ctypes.c_double * point_count)(
                    *self.storage.point_charge_values[item.point_begin : item.point_end]
                )
                point_positions = (ctypes.c_double * (3 * point_count))(
                    *self.storage.point_charge_positions[
                        3 * item.point_begin : 3 * item.point_end
                    ]
                )
                npoints = ctypes.c_int(point_count)
                self.library.xtb_setExternalCharges(
                    environment,
                    calculator,
                    ctypes.byref(npoints),
                    point_numbers,
                    point_charges,
                    point_positions,
                )
                self._check(environment, "xtb_setExternalCharges")
            gradient = (
                (ctypes.c_double * (3 * atom_count))()
                if self.property_name == "force"
                else None
            )
            point_gradient = (
                (ctypes.c_double * (3 * point_count))()
                if self.property_name == "force" and point_count
                else None
            )
            return XtbState(
                environment=environment,
                molecule=molecule,
                calculator=calculator,
                result=result,
                positions=positions,
                energy=ctypes.c_double(),
                gradient=gradient,
                point_gradient=point_gradient,
                has_external_charges=bool(point_count),
                point_count=npoints if point_count else None,
                point_numbers=point_numbers if point_count else None,
                point_charges=point_charges if point_count else None,
                point_positions=point_positions if point_count else None,
                keepalive=(
                    atomic_numbers,
                    charge,
                    unpaired,
                    natoms,
                    *(
                        (point_numbers, point_charges, point_positions, npoints)
                        if point_count
                        else ()
                    ),
                ),
            )
        except BaseException:
            _delete(self.library, "xtb_delResults", result)
            _delete(self.library, "xtb_delCalculator", calculator)
            _delete(self.library, "xtb_delMolecule", molecule)
            _delete(self.library, "xtb_delEnvironment", environment)
            raise

    def invoke(self) -> None:
        """Run a serial logical batch, including geometry update and result getters."""
        for state in self.states:
            self.library.xtb_updateMolecule(
                state.environment, state.molecule, state.positions, None
            )
            self._check(state.environment, "xtb_updateMolecule")
            if state.has_external_charges:
                # libxtb 6.7.1 accumulates the PC gradient when the same
                # external-charge object is reused. Rebinding the persistent
                # caller arrays resets that per-call result while retaining
                # environment, molecule, calculator, result, and SCC restart.
                self.library.xtb_releaseExternalCharges(
                    state.environment, state.calculator
                )
                self._check(state.environment, "xtb_releaseExternalCharges")
                self.library.xtb_setExternalCharges(
                    state.environment,
                    state.calculator,
                    ctypes.byref(state.point_count),
                    state.point_numbers,
                    state.point_charges,
                    state.point_positions,
                )
                self._check(state.environment, "xtb_setExternalCharges")
            self.library.xtb_singlepoint(
                state.environment, state.molecule, state.calculator, state.result
            )
            self._check(state.environment, "xtb_singlepoint")
            self.library.xtb_getEnergy(
                state.environment, state.result, ctypes.byref(state.energy)
            )
            self._check(state.environment, "xtb_getEnergy")
            if state.gradient is not None:
                # xTB's getter ABI permits additive publication (notably for
                # PC gradients), so callers must provide zeroed destinations.
                ctypes.memset(
                    ctypes.addressof(state.gradient), 0, ctypes.sizeof(state.gradient)
                )
                self.library.xtb_getGradient(
                    state.environment, state.result, state.gradient
                )
                self._check(state.environment, "xtb_getGradient")
            if state.point_gradient is not None:
                ctypes.memset(
                    ctypes.addressof(state.point_gradient),
                    0,
                    ctypes.sizeof(state.point_gradient),
                )
                self.library.xtb_getPCGradient(
                    state.environment, state.result, state.point_gradient
                )
                self._check(state.environment, "xtb_getPCGradient")

    def results(self) -> dict[str, Any]:
        """Normalize xTB gradients to gpuxtb's public force convention."""
        output: dict[str, Any] = {
            "energies_hartree": [float(state.energy.value) for state in self.states]
        }
        if self.property_name == "force":
            output["forces_hartree_per_bohr"] = [
                -float(value)
                for state in self.states
                for value in (state.gradient or ())
            ]
            if any(state.point_gradient is not None for state in self.states):
                output["point_charge_forces_hartree_per_bohr"] = [
                    -float(value)
                    for state in self.states
                    for value in (state.point_gradient or ())
                ]
        return output

    def close(self) -> None:
        """Release all persistent states in reverse construction order."""
        while self.states:
            state = self.states.pop()
            if state.has_external_charges:
                self.library.xtb_releaseExternalCharges(
                    state.environment, state.calculator
                )
                # Cleanup continues even if release recorded a diagnostic.
            _delete(self.library, "xtb_delResults", state.result)
            _delete(self.library, "xtb_delCalculator", state.calculator)
            _delete(self.library, "xtb_delMolecule", state.molecule)
            _delete(self.library, "xtb_delEnvironment", state.environment)
