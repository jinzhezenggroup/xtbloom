"""Persistent in-process adapter for the PyTorch dxtb GFN2 implementation.

The adapter consumes the same ``PublicBatchStorage`` used by the gpuxtb
benchmark.  Coordinates are already expressed in bohr and dxtb reports
energies in Hartree, so no unit conversion is performed.  Reported forces are
therefore in Hartree/bohr and already use the negative-gradient convention.

dxtb memoizes calculator properties using input tensor identity.  Returning a
memoized value would not be an inference benchmark, so :meth:`invoke` includes
``Calculator.reset()`` in every measured call.  The comparatively expensive
calculator construction, parameter materialization, batch index helper, and
input tensor allocation remain persistent setup state.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Self

_LOCAL_DXTB_SOURCE_ROOT = Path.home() / "codes" / "dxtb"
DEFAULT_DXTB_SOURCE_ROOT: Path | None = (
    _LOCAL_DXTB_SOURCE_ROOT if _LOCAL_DXTB_SOURCE_ROOT.is_dir() else None
)
SUPPORTED_BACKENDS = {"cpu", "cuda"}
SUPPORTED_PROPERTIES = {"energy", "force"}


class DxtbError(RuntimeError):
    """An invalid benchmark request or failing dxtb runtime operation."""


def _import_runtime(
    source_root: Path | None,
) -> tuple[ModuleType, ModuleType]:
    """Import torch and dxtb, optionally preferring a source checkout.

    A source checkout is added to ``sys.path`` only when ``dxtb`` has not
    already been imported.  This keeps the benchmark in-process while making
    the exact checkout at ``~/codes/dxtb`` usable without installing dxtb
    itself.  dxtb's normal Python dependencies, including ``tad-libcint`` for
    GFN2, must still be installed in the active environment.
    """

    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DxtbError(f"PyTorch is unavailable: {exc}") from exc

    if source_root is not None and "dxtb" not in sys.modules:
        source_directory = source_root.expanduser().resolve() / "src"
        package = source_directory / "dxtb" / "__init__.py"
        if not package.is_file():
            raise DxtbError(
                f"dxtb source checkout is missing package entry point: {package}"
            )
        source_text = str(source_directory)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

    try:
        dxtb = importlib.import_module("dxtb")
    except ImportError as exc:  # pragma: no cover - environment dependent
        hint = (
            "install dxtb[libcint] and its dependencies, or provide a dxtb "
            "source checkout whose dependencies are installed"
        )
        raise DxtbError(f"dxtb is unavailable ({exc}); {hint}") from exc

    if not hasattr(dxtb, "Calculator") or not hasattr(dxtb, "GFN2_XTB"):
        raise DxtbError("imported dxtb module lacks Calculator or GFN2_XTB")
    return torch, dxtb


class DxtbAdapter:
    """Run one persistent, homogeneous-or-padded dxtb GFN2 batch.

    Parameters
    ----------
    storage
        A ``PublicBatchStorage``-compatible object.  Atomic positions must be
        flattened atom-major coordinates in bohr.
    property_name
        ``"energy"`` or ``"force"``.  Force inference uses PyTorch autograd
        over one dxtb single-point result, so energy and force are produced by
        the same SCC calculation.
    backend
        ``"cpu"`` or ``"cuda"``.  CUDA input, model state, and outputs stay
        device resident until :meth:`results`, which is intentionally outside
        the measured interval.
    device_id
        CUDA device index.  Ignored for CPU execution.
    cpu_threads
        Intra-op PyTorch thread count.  The previous process setting is
        restored by :meth:`close`.
    source_root
        Optional dxtb checkout.  Defaults to ``~/codes/dxtb`` when present;
        pass ``None`` to use an already-installed package without source
        discovery.

    Notes
    -----
    Repeated identical atomic-number rows use dxtb conformer batch mode 2.
    Other ragged rows are zero-padded and use batch mode 1.  Padding is removed
    from published forces, preserving the original public batch atom order.
    """

    external_point_charge_reason = (
        "dxtb GFN2 benchmark adapter does not yet map gpuxtb's discrete "
        "external point-charge SCC and point-charge-force contract"
    )

    def __init__(
        self,
        storage: Any,
        property_name: str,
        backend: str,
        device_id: int = 0,
        cpu_threads: int = 1,
        source_root: Path | None = DEFAULT_DXTB_SOURCE_ROOT,
        accuracy: float = 1.0e-4,
        max_iterations: int = 500,
        *,
        _runtime: tuple[ModuleType, ModuleType] | None = None,
    ) -> None:
        if property_name not in SUPPORTED_PROPERTIES:
            raise DxtbError(f"unsupported dxtb property: {property_name}")
        if backend not in SUPPORTED_BACKENDS:
            raise DxtbError(f"unsupported dxtb backend: {backend}")
        if type(device_id) is not int or device_id < 0:
            raise DxtbError("dxtb device_id must be a nonnegative integer")
        if type(cpu_threads) is not int or cpu_threads <= 0:
            raise DxtbError("dxtb cpu_threads must be a positive integer")
        if accuracy <= 0.0:
            raise DxtbError("dxtb SCC accuracy must be positive")
        if type(max_iterations) is not int or max_iterations <= 0:
            raise DxtbError("dxtb max_iterations must be a positive integer")
        if getattr(storage, "point_charge_values", None):
            raise DxtbError(self.external_point_charge_reason)

        self.storage = storage
        self.property_name = property_name
        self.backend = backend
        self.device_id = device_id
        self.cpu_threads = cpu_threads
        self.accuracy = float(accuracy)
        self.max_iterations = max_iterations
        self.torch, self.dxtb = (
            _runtime if _runtime is not None else _import_runtime(source_root)
        )
        self._closed = False
        self._energies: Any | None = None
        self._forces: Any | None = None
        self._calculator: Any | None = None
        self._previous_threads: int | None = None
        self._timer: Any | None = getattr(self.dxtb, "timer", None)
        self._previous_timer_cuda_sync: bool | None = None

        if backend == "cuda":
            if not self.torch.cuda.is_available():
                raise DxtbError(
                    "dxtb CUDA backend requested but torch.cuda is unavailable"
                )
            device_count = int(self.torch.cuda.device_count())
            if device_id >= device_count:
                raise DxtbError(
                    f"dxtb CUDA device {device_id} is outside available range "
                    f"[0, {device_count})"
                )
            self.device = self.torch.device(f"cuda:{device_id}")
        else:
            self.device = self.torch.device("cpu")

        slices = list(getattr(storage, "slices", ()))
        if not slices:
            raise DxtbError("dxtb benchmark batch must contain at least one system")
        self.batch_size = len(slices)
        self.atom_counts: list[int] = []
        number_rows: list[list[int]] = []
        position_rows: list[list[list[float]]] = []
        total_atoms = len(storage.atomic_numbers)
        if len(storage.positions) != 3 * total_atoms:
            raise DxtbError(
                "dxtb storage positions length must equal three times total atoms"
            )
        for index, item in enumerate(slices):
            begin = int(item.atom_begin)
            end = int(item.atom_end)
            if begin < 0 or end <= begin or end > total_atoms:
                raise DxtbError(f"dxtb system {index} has invalid atom slice")
            count = end - begin
            self.atom_counts.append(count)
            number_rows.append(
                [int(value) for value in storage.atomic_numbers[begin:end]]
            )
            flat_positions = storage.positions[3 * begin : 3 * end]
            position_rows.append(
                [
                    [float(flat_positions[3 * atom + axis]) for axis in range(3)]
                    for atom in range(count)
                ]
            )

        if len(storage.molecular_charges) != self.batch_size:
            raise DxtbError("dxtb storage must provide one charge per system")
        if len(storage.unpaired_electrons) != self.batch_size:
            raise DxtbError("dxtb storage must provide one spin per system")

        self.max_atoms = max(self.atom_counts)
        conformer_batch = all(
            row == number_rows[0] and len(row) == self.max_atoms for row in number_rows
        )
        self.batch_mode = 2 if conformer_batch else 1
        padded_numbers: list[list[int]] = []
        padded_positions: list[list[list[float]]] = []
        self._force_flat_indices: list[int] = []
        for system, (numbers, positions) in enumerate(zip(number_rows, position_rows)):
            padding = self.max_atoms - len(numbers)
            padded_numbers.append(numbers + [0] * padding)
            padded_positions.append(positions + [[0.0, 0.0, 0.0]] * padding)
            for atom in range(len(numbers)):
                base = 3 * (system * self.max_atoms + atom)
                self._force_flat_indices.extend((base, base + 1, base + 2))

        try:
            self._previous_threads = int(self.torch.get_num_threads())
            self.torch.set_num_threads(cpu_threads)
            if self._timer is not None and hasattr(self._timer, "cuda_sync"):
                self._previous_timer_cuda_sync = bool(self._timer.cuda_sync)
                # dxtb's diagnostic timer can otherwise synchronize individual
                # CUDA stages and distort an outer end-to-end timing boundary.
                self._timer.cuda_sync = False

            self.numbers = self.torch.tensor(
                padded_numbers, dtype=self.torch.long, device=self.device
            )
            self.positions = self.torch.tensor(
                padded_positions, dtype=self.torch.float64, device=self.device
            )
            if property_name == "force":
                self.positions.requires_grad_(True)
            self.charges = self.torch.tensor(
                [float(value) for value in storage.molecular_charges],
                dtype=self.torch.float64,
                device=self.device,
            )
            self.spins = self.torch.tensor(
                [float(value) for value in storage.unpaired_electrons],
                dtype=self.torch.float64,
                device=self.device,
            )
            self.options = {
                "verbosity": 0,
                "batch_mode": self.batch_mode,
                "maxiter": max_iterations,
                # dxtb exposes the SCF fixed-point/function tolerances directly;
                # using the same value for both is its closest in-process
                # equivalent to the cross-library ``--acc 0.0001`` contract.
                "x_atol": self.accuracy,
                "f_atol": self.accuracy,
            }
            self._calculator = self.dxtb.Calculator(
                self.numbers,
                self.dxtb.GFN2_XTB,
                opts=self.options,
                device=self.device,
                dtype=self.torch.float64,
            )
        except BaseException as exc:
            self.close()
            if isinstance(exc, DxtbError):
                raise
            raise DxtbError(
                f"failed to construct persistent dxtb GFN2 batch: {exc}"
            ) from exc

        self.version = getattr(self.dxtb, "__version__", None)
        self.torch_version = getattr(self.torch, "__version__", None)
        self.module_path = getattr(self.dxtb, "__file__", None)
        self.thread_control = {
            "requested_torch_threads": cpu_threads,
            "previous_torch_threads": self._previous_threads,
            "torch_interop_threads": (
                int(self.torch.get_num_interop_threads())
                if hasattr(self.torch, "get_num_interop_threads")
                else None
            ),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        }

    @property
    def calculator(self) -> Any:
        """Return the live calculator while rejecting use after cleanup."""
        if self._closed or self._calculator is None:
            raise DxtbError("dxtb adapter is closed")
        return self._calculator

    def invoke(self) -> None:
        """Execute one real GFN2 SCC inference without host publication.

        Resetting dxtb's identity-keyed result/component caches is deliberately
        part of this call.  On CUDA, kernel completion remains asynchronous
        until :meth:`synchronize`, allowing the harness to place a precise
        completion boundary around the measured operation.
        """

        calculator = self.calculator
        self._energies = None
        self._forces = None
        try:
            calculator.reset()
            result = calculator.singlepoint(
                self.positions,
                self.charges,
                self.spins,
                cuda_sync_in_scf=False,
            )
            energies = result.total.sum(dim=-1)
            if int(energies.numel()) != self.batch_size:
                raise DxtbError(
                    "dxtb returned an energy shape inconsistent with the logical batch"
                )
            if self.property_name == "force":
                (gradient,) = self.torch.autograd.grad(
                    energies.sum(),
                    self.positions,
                    create_graph=False,
                    retain_graph=False,
                )
                self._forces = (-gradient).detach()
            self._energies = energies.detach()
        except BaseException as exc:
            self._energies = None
            self._forces = None
            if isinstance(exc, DxtbError):
                raise
            raise DxtbError(f"dxtb GFN2 inference failed: {exc}") from exc

    def synchronize(self) -> None:
        """Complete all CUDA work submitted by :meth:`invoke`; CPU is synchronous."""

        if self._closed:
            raise DxtbError("dxtb adapter is closed")
        if self.backend == "cuda":
            self.torch.cuda.synchronize(self.device)

    def results(self) -> dict[str, Any]:
        """Copy synchronized outputs to Python lists in public gpuxtb units."""

        if self._energies is None:
            raise DxtbError("dxtb results requested before a successful invoke")
        self.synchronize()
        energy_values = self._energies.to(device="cpu").reshape(-1).tolist()
        output: dict[str, Any] = {
            "energies_hartree": [float(value) for value in energy_values]
        }
        if self.property_name == "force":
            if self._forces is None:
                raise DxtbError("dxtb force inference did not publish forces")
            padded = self._forces.to(device="cpu").reshape(-1)
            output["forces_hartree_per_bohr"] = [
                float(padded[index].item()) for index in self._force_flat_indices
            ]
        return output

    def close(self) -> None:
        """Release tensors/calculator and restore process-global torch controls."""

        if self._closed:
            return
        self._closed = True
        self._forces = None
        self._energies = None
        self._calculator = None
        for name in ("numbers", "positions", "charges", "spins"):
            if hasattr(self, name):
                setattr(self, name, None)
        if (
            self._timer is not None
            and self._previous_timer_cuda_sync is not None
            and hasattr(self._timer, "cuda_sync")
        ):
            self._timer.cuda_sync = self._previous_timer_cuda_sync
        if self._previous_threads is not None:
            self.torch.set_num_threads(self._previous_threads)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.close()
        return False


def timed_invoke(adapter: DxtbAdapter) -> float:
    """Measure one inference through an explicit CPU/CUDA completion boundary."""

    start = time.perf_counter_ns()
    adapter.invoke()
    adapter.synchronize()
    return (time.perf_counter_ns() - start) * 1.0e-6
