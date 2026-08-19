"""Model-aware packed Array API/DLPack inference."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np

from . import _dlpack, library
from . import interface as _interface

if TYPE_CHECKING:
    from types import TracebackType


class ArrayBatch(_interface.ArrayBatch):
    """Packed ragged-batch GFN1/GFN2 inference over Array API/DLPack arrays.

    The descriptor, ownership, copy, stream, output, and failure semantics are
    identical to :class:`xtbloom.interface.ArrayBatch`. ``method`` selects the
    public xTB model and defaults to GFN2-xTB for backward compatibility.
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
        method: str = "GFN2-xTB",
        copy: bool = False,
        backend: str | int = "auto",
        device_id: int | None = None,
        cpu_threads: int = 1,
        stream: int | None = None,
    ) -> None:
        self._model = _interface._resolve_method(method)
        self._method = method
        resolved_backend = _interface._backend_for_model(
            self._model, backend, device_id
        )
        super().__init__(
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
            backend=resolved_backend,
            device_id=device_id,
            cpu_threads=cpu_threads,
            stream=stream,
        )

    @property
    def method(self) -> str:
        """Return the configured GFN method name."""
        return self._method

    def compute(
        self,
        *,
        max_scc_iterations: int = 250,
        charge_tolerance: float = 1.0e-6,
        energy_tolerance: float = 1.0e-8,
        electronic_temperature: float = 300.0,
        scc_mixer: str | int = "modified_broyden",
        scc_mixer_history: int = library.DEFAULT_SCC_MIXER_HISTORY,
        scc_mixer_damping: float = library.DEFAULT_SCC_MIXER_DAMPING,
        determinism: str | int = "default",
        compute_energy: bool = True,
        compute_forces: bool = True,
        compute_charges: bool = True,
        compute_point_charge_forces: bool | None = None,
        out: object | None = None,
        result_memory: str = "host",
    ) -> _interface.ArrayBatchResult:
        """Run one synchronous packed calculation with the selected model."""
        self._context._create()
        return _compute_array_batch(
            self,
            max_scc_iterations=max_scc_iterations,
            charge_tolerance=charge_tolerance,
            energy_tolerance=energy_tolerance,
            electronic_temperature=electronic_temperature,
            scc_mixer=scc_mixer,
            scc_mixer_history=scc_mixer_history,
            scc_mixer_damping=scc_mixer_damping,
            determinism=determinism,
            compute_energy=compute_energy,
            compute_forces=compute_forces,
            compute_charges=compute_charges,
            compute_point_charge_forces=compute_point_charge_forces,
            out=out,
            result_memory=result_memory,
        )

    def __enter__(self) -> ArrayBatch:  # noqa: PYI034 - 3.10 lacks Self
        """Return this batch for use in a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the native context on context-manager exit."""
        self.close()


def _compute_array_batch(
    batch: ArrayBatch,
    *,
    max_scc_iterations: int,
    charge_tolerance: float,
    energy_tolerance: float,
    electronic_temperature: float,
    scc_mixer: str | int,
    scc_mixer_history: int,
    scc_mixer_damping: float,
    determinism: str | int,
    compute_energy: bool,
    compute_forces: bool,
    compute_charges: bool,
    compute_point_charge_forces: bool | None,
    out: object | None,
    result_memory: str,
) -> _interface.ArrayBatchResult:
    """Reuse the packed descriptor machinery with a model-aware option tag."""
    if result_memory not in ("host", "cuda"):
        raise _interface.XTBloomValueError(
            f"result_memory must be 'host' or 'cuda', got {result_memory!r}"
        )
    if result_memory == "cuda" and int(batch.backend) != library.BACKEND_CUDA:
        raise _interface.XTBloomNotSupportedError(
            "result_memory='cuda' requires the resolved CUDA backend; "
            "run a CUDA ArrayBatch or use result_memory='host'"
        )
    arrays = batch._arrays
    context = batch._context
    context._create()
    nsystems, natoms, npoints, response_elements = _interface._derive_batch_counts(
        arrays
    )
    out_spec = _interface._normalize_out_spec(out)
    requested_arrays = [array for array in arrays.values() if array is not None]
    requested_arrays.extend(out_spec.values())
    _interface._validate_array_devices_before_export(requested_arrays, context)

    views: list[_dlpack.DLPackView] = []
    keepalive: list[object] = []
    output_owners: dict[str, object] = {}
    arenas: list[_dlpack._ResultArena] = []
    xtbloom_owned: list[_dlpack.DLPackResultBuffer] = []
    committed_result: _interface.ArrayBatchResult | None = None
    try:
        descriptor = library.Batch()
        library._check_init(
            "xtbloom_batch_init",
            library.load_library().xtbloom_batch_init(
                ctypes.byref(descriptor), ctypes.sizeof(descriptor)
            ),
        )
        descriptor.batch_size = nsystems
        descriptor.total_atoms = natoms
        descriptor.total_point_charges = npoints
        descriptor.total_charge_response_elements = response_elements

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
                expected_cuda_device=context.device_id,
                copy=batch._copy,
            )
            views.append(view)
            setattr(descriptor, name, view.descriptor)
            return view

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

        _interface._validate_device_consistency(views, context)
        options = _interface._build_compute_options(
            nsystems,
            npoints,
            max_scc_iterations=max_scc_iterations,
            charge_tolerance=charge_tolerance,
            energy_tolerance=energy_tolerance,
            electronic_temperature=electronic_temperature,
            scc_mixer=scc_mixer,
            scc_mixer_history=scc_mixer_history,
            scc_mixer_damping=scc_mixer_damping,
            determinism=determinism,
            compute_energy=compute_energy,
            compute_forces=compute_forces,
            compute_charges=compute_charges,
            compute_point_charge_forces=compute_point_charge_forces,
        )
        options.model = batch._model
        _interface._validate_requested_outputs(out_spec, options.flags, npoints)
        result = library.BatchResult()
        library._check_init(
            "xtbloom_batch_result_init",
            library.load_library().xtbloom_batch_result_init(
                ctypes.byref(result), ctypes.sizeof(result)
            ),
        )
        _interface._bind_outputs(
            result,
            out_spec,
            views,
            keepalive,
            output_owners,
            context.stream,
            context.device_id,
            nsystems,
            natoms,
            npoints,
            flags=options.flags,
            result_memory=result_memory,
            context=context,
            arenas=arenas,
            xtbloom_owned=xtbloom_owned,
        )
        _interface._validate_device_consistency(views, context)
        library.compute_checked(context._create(), descriptor, options, result)
        array_result = _interface.ArrayBatchResult(
            output_owners, result_flags=int(result.flags)
        )
        array_result._attach_producers(arenas, xtbloom_owned)
        committed_result = array_result
        return array_result
    finally:
        _dlpack.release_all(views)
        if committed_result is None:
            for producer in xtbloom_owned:
                producer.close()
            for arena in arenas:
                arena.close()


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
    method: str = "GFN2-xTB",
    copy: bool = False,
    backend: str | int = "auto",
    device_id: int | None = None,
    cpu_threads: int = 1,
    stream: int | None = None,
    max_scc_iterations: int = 250,
    charge_tolerance: float = 1.0e-6,
    energy_tolerance: float = 1.0e-8,
    electronic_temperature: float = 300.0,
    scc_mixer: str | int = "modified_broyden",
    scc_mixer_history: int = library.DEFAULT_SCC_MIXER_HISTORY,
    scc_mixer_damping: float = library.DEFAULT_SCC_MIXER_DAMPING,
    determinism: str | int = "default",
    out: object | None = None,
    result_memory: str = "host",
) -> _interface.ArrayBatchResult:
    """One-shot model-aware packed inference."""
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
        method=method,
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
            scc_mixer=scc_mixer,
            scc_mixer_history=scc_mixer_history,
            scc_mixer_damping=scc_mixer_damping,
            determinism=determinism,
            out=out,
            result_memory=result_memory,
        )


__all__ = ["ArrayBatch", "compute_arrays"]
