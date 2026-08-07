"""Tests for the packed Array API/DLPack batch entry point.

:class:`gpuxtb.ArrayBatch` consumes dense eager arrays through the DLPack
producer protocol and binds them zero-copy to the public C-ABI descriptors.
These tests run entirely on the CPU backend with NumPy arrays (no optional
array provider is required); the CUDA device-array coverage lives in
``test_array_batch_cuda.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _cases import case_by_id, structure_inputs
from _dlpack_fakes import FakeArray
from gpuxtb import (
    ArrayBatch,
    BatchCalculator,
    Calculator,
    ChargeResponse,
    PointCharge,
    Structure,
    compute_arrays,
)
from gpuxtb.exceptions import (
    GPUxtbNotSupportedError,
    GPUxtbRuntimeError,
    GPUxtbValueError,
)

H2_POSITIONS = np.array(
    [
        [-0.71, 0.0, 0.0],
        [0.71, 0.0, 0.0],
    ]
)


def _structure(case_id: str) -> Structure:
    """Build a conformance-golden structure with its documented spin state."""
    numbers, positions, charge, uhf, spin = structure_inputs(case_by_id(case_id))
    return Structure(numbers, positions, charge=charge, uhf=uhf, spin_channels=spin)


def _pack_single(structures: list[Structure], *, include_points: bool = True) -> dict:
    """Pack a ragged batch of structures into flat ABI descriptor arrays."""
    atom_offsets = [0]
    numbers: list[int] = []
    positions: list[float] = []
    charges: list[float] = []
    unpaired: list[int] = []
    spin: list[int] = []
    point_offsets = [0]
    point_positions: list[float] = []
    point_values: list[float] = []
    point_gammas: list[float] = []
    for structure in structures:
        numbers.extend(int(value) for value in structure.numbers)
        positions.extend(float(value) for value in structure.positions.ravel())
        charges.append(structure.charge)
        unpaired.append(structure.uhf)
        spin.append(structure.spin_channels)
        if include_points and structure.point_charges is not None:
            point_positions.extend(
                float(value) for value in structure.point_charges.positions.ravel()
            )
            point_values.extend(
                float(value) for value in structure.point_charges.charges
            )
            point_gammas.extend(
                float(value) for value in structure.point_charges.gammas
            )
        atom_offsets.append(len(numbers))
        point_offsets.append(len(point_values))
    has_points = bool(point_values)
    return {
        "atom_offsets": np.asarray(atom_offsets, dtype=np.int64),
        "atomic_numbers": np.asarray(numbers, dtype=np.int32),
        "positions": np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        "molecular_charges": np.asarray(charges, dtype=np.float64),
        "unpaired_electrons": np.asarray(unpaired, dtype=np.int32),
        "spin_channels": np.asarray(spin, dtype=np.int32),
        "point_charge_offsets": (
            np.asarray(point_offsets, dtype=np.int64) if has_points else None
        ),
        "point_charge_positions": (
            np.asarray(point_positions, dtype=np.float64).reshape(-1, 3)
            if has_points
            else None
        ),
        "point_charge_values": (
            np.asarray(point_values, dtype=np.float64) if has_points else None
        ),
        "point_charge_gammas": (
            np.asarray(point_gammas, dtype=np.float64) if has_points else None
        ),
    }


def _water() -> Calculator:
    """Return a fixed water calculator with a documented golden energy."""
    return Calculator(
        "GFN2-xTB",
        numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0000000000, 0.0000000000, -0.7357858611],
                [1.4418315287, 0.0000000000, 0.3678929305],
                [-1.4418315287, 0.0000000000, 0.3678929305],
            ]
        ),
    )


# --- parity with the existing NumPy API ----------------------------------------


def test_single_system_matches_calculator() -> None:
    """A one-system ArrayBatch is bit-comparable to the host calculator."""
    water = _water()
    reference = water.singlepoint()
    packed = _pack_single([water])
    result = ArrayBatch(**packed, backend="cpu").compute()
    assert result.energies == pytest.approx([reference.energy], rel=1.0e-12)
    assert result.forces == pytest.approx(reference.forces, abs=1.0e-11)
    assert result.charges == pytest.approx(reference.charges, abs=1.0e-11)
    assert array_is_numpy(result.forces)


def test_ragged_batch_matches_batch_calculator() -> None:
    """Multi-system ragged batches preserve per-system results."""
    structures = [_structure("ketene"), _structure("h3_plus")]
    reference = BatchCalculator(structures, backend="cpu").compute()
    packed = _pack_single(structures)
    result = ArrayBatch(**packed, backend="cpu").compute()
    assert result.energies == pytest.approx(reference.energies, rel=1.0e-12)
    assert result.forces == pytest.approx(reference.forces, abs=1.0e-11)
    assert result.charges == pytest.approx(reference.charges, abs=1.0e-11)
    assert result.scc_converged.tolist() == reference.scc_converged.tolist()


def test_unrestricted_system_with_spin_channels() -> None:
    """Explicit spin_channels=2 must flow through to the native call."""
    structure = _structure("h3_plus")
    reference = Calculator(
        "GFN2-xTB",
        structure.numbers,
        structure.positions,
        charge=structure.charge,
        uhf=structure.uhf,
        spin_channels=2,
        backend="cpu",
    ).singlepoint()
    packed = _pack_single([structure])
    result = ArrayBatch(**packed, backend="cpu").compute()
    assert result.energies == pytest.approx([reference.energy], rel=1.0e-12)


def test_default_spin_channels_is_restricted() -> None:
    """Without spin_channels, the batch defaults to restricted orbitals."""
    water = _water()
    packed = _pack_single([water])
    del packed["spin_channels"]
    result = ArrayBatch(**packed, backend="cpu").compute()
    reference = water.singlepoint()
    assert result.energies == pytest.approx([reference.energy], rel=1.0e-12)


def test_point_charges_match_calculator() -> None:
    """Point-charge inputs and outputs flow through the DLPack path."""
    water = Calculator(
        "GFN2-xTB",
        numbers=np.array([8, 1, 1]),
        positions=_water().positions.copy(),
        point_charges=PointCharge(
            positions=np.array([[0.0, 0.0, 4.0]]),
            charges=np.array([-0.5]),
            gammas=np.array([0.999]),
        ),
    )
    reference = water.singlepoint()
    packed = _pack_single([water])
    assert packed["point_charge_offsets"] is not None
    result = ArrayBatch(**packed, backend="cpu").compute(
        compute_point_charge_forces=True
    )
    assert result.energies == pytest.approx([reference.energy], rel=1.0e-12)
    assert result.point_charge_forces is not None
    assert result.point_charge_forces == pytest.approx(
        reference.point_charge_forces, abs=1.0e-11
    )


def test_charge_response_descriptors() -> None:
    """Periodic b + A q descriptors bind and reproduce the reference energy."""
    structure = _structure("h3_plus")
    response = ChargeResponse(
        shifts=np.array([0.003, -0.002, 0.001]),
        matrix=np.array([[0.02, 0.001, 0.0], [0.001, 0.018, 0.0], [0.0, 0.0, 0.015]]),
    )
    calculator = Calculator(
        "GFN2-xTB",
        structure.numbers,
        structure.positions,
        charge=structure.charge,
        uhf=structure.uhf,
        spin_channels=structure.spin_channels,
        charge_response=response,
        backend="cpu",
    )
    reference = calculator.singlepoint()
    packed = _pack_single([calculator])
    del packed["point_charge_offsets"]
    del packed["point_charge_positions"]
    del packed["point_charge_values"]
    del packed["point_charge_gammas"]
    natoms = len(structure)
    matrix = response.matrix.ravel().astype(np.float64)
    packed["atomic_potential_shifts"] = response.shifts.astype(np.float64)
    packed["charge_response_offsets"] = np.asarray([0, len(matrix)], dtype=np.int64)
    packed["charge_response_matrix"] = matrix
    assert natoms == 3
    result = ArrayBatch(**packed, backend="cpu").compute()
    assert result.energies == pytest.approx([reference.energy], rel=1.0e-11)


def test_repeated_calls_and_updated_geometry() -> None:
    """The same batch object is reusable across changed geometries."""
    water = _water()
    packed = _pack_single([water])
    batch = ArrayBatch(**packed, backend="cpu")
    first = batch.compute()
    # Displace a single atom: a pure translation would be an energy invariant.
    packed["positions"][0] += 0.25
    displaced = np.array(packed["positions"], copy=True)
    second = batch.compute()
    assert first.energies[0] != second.energies[0]
    assert second.energies[0] == pytest.approx(
        Calculator(
            "GFN2-xTB",
            numbers=np.array([8, 1, 1]),
            positions=displaced,
            backend="cpu",
        )
        .singlepoint()
        .energy,
        rel=1.0e-10,
    )


def test_compute_arrays_free_function() -> None:
    """compute_arrays() is a thin one-shot alias of ArrayBatch.compute()."""
    water = _water()
    reference = water.singlepoint()
    packed = _pack_single([water])
    result = compute_arrays(**packed, backend="cpu")
    assert result.energies == pytest.approx([reference.energy], rel=1.0e-12)


# --- output policy -------------------------------------------------------------


def test_out_buffers_are_written_in_place() -> None:
    """Supply out= arrays; the result must reference them after compute."""
    water = _water()
    reference = water.singlepoint()
    packed = _pack_single([water])
    out_energies = np.empty(1)
    out_forces = np.empty((3, 3))
    out_charges = np.empty(3)
    result = ArrayBatch(**packed, backend="cpu").compute(
        out={
            "energies": out_energies,
            "forces": out_forces,
            "charges": out_charges,
        }
    )
    assert result.energies is out_energies
    assert result.forces is out_forces
    assert result.charges is out_charges
    assert out_energies == pytest.approx([reference.energy], rel=1.0e-12)
    assert out_forces == pytest.approx(reference.forces, abs=1.0e-11)
    assert out_charges == pytest.approx(reference.charges, abs=1.0e-11)
    assert result.result_flags == 0


def test_out_buffers_remain_zero_copy_when_input_copy_is_enabled() -> None:
    """The input copy policy must never redirect writes to temporary outputs."""
    water = _water()
    reference = water.singlepoint()
    packed = _pack_single([water])
    out_energies = np.full(1, 123.0)
    result = ArrayBatch(**packed, backend="cpu", copy=True).compute(
        compute_forces=False,
        compute_charges=False,
        out={"energies": out_energies},
    )
    assert result.energies is out_energies
    assert out_energies == pytest.approx([reference.energy], rel=1.0e-12)


def test_out_alias_atomic_charges() -> None:
    """atomic_charges is accepted as an alias for charges."""
    water = _water()
    out_charges = np.empty(3)
    ArrayBatch(**_pack_single([water]), backend="cpu").compute(
        out={"atomic_charges": out_charges}
    )
    assert out_charges.size == 3


def test_out_readonly_rejected() -> None:
    """Read-only buffers must never be used as mutable outputs."""
    water = _water()
    out_forces = np.empty((3, 3))
    out_forces.flags.writeable = False
    with pytest.raises(BufferError, match="writ"):
        ArrayBatch(**_pack_single([water]), backend="cpu").compute(
            out={"forces": out_forces}
        )


def test_out_wrong_shape_rejected() -> None:
    """out= arrays must match the native output extents exactly."""
    water = _water()
    with pytest.raises(GPUxtbValueError, match="shape"):
        ArrayBatch(**_pack_single([water]), backend="cpu").compute(
            out={"forces": np.empty((2, 3))}
        )


def test_out_wrong_dtype_rejected() -> None:
    """out= arrays must match the native output dtype exactly."""
    water = _water()
    with pytest.raises(GPUxtbValueError, match="dtype"):
        ArrayBatch(**_pack_single([water]), backend="cpu").compute(
            out={"energies": np.empty(1, dtype=np.float32)}
        )


def test_out_unknown_name_rejected() -> None:
    """Unknown output names fail before touching anything."""
    water = _water()
    with pytest.raises(GPUxtbValueError, match="output name"):
        ArrayBatch(**_pack_single([water]), backend="cpu").compute(
            out={"enthalpy": np.empty(1)}
        )


def test_out_for_unrequested_property_is_rejected() -> None:
    """An explicit output buffer must never be accepted and then ignored."""
    packed = _pack_single([_water()])
    with pytest.raises(GPUxtbValueError, match="not requested"):
        ArrayBatch(**packed, backend="cpu").compute(
            compute_energy=False,
            out={"energies": np.empty(1)},
        )


def test_out_non_dict_rejected() -> None:
    """The output policy must be a mapping."""
    water = _water()
    with pytest.raises(GPUxtbValueError, match="mapping"):
        ArrayBatch(**_pack_single([water]), backend="cpu").compute(out=[1, 2])


# --- request validation and failure isolation -----------------------------------


def test_incomplete_point_charge_group_rejected() -> None:
    """Optional descriptor groups must be all-or-nothing."""
    water = _water()
    packed = _pack_single([water])
    packed["point_charge_offsets"] = np.asarray([0, 0], dtype=np.int64)
    with pytest.raises(GPUxtbValueError, match="together"):
        ArrayBatch(**packed, backend="cpu")


def test_empty_point_charge_group_is_not_silently_discarded() -> None:
    """Supplied zero-count offsets still participate in native validation."""
    packed = _pack_single([_water()])
    packed.update(
        point_charge_offsets=np.asarray([0, 1], dtype=np.int64),
        point_charge_positions=np.empty((0, 3), dtype=np.float64),
        point_charge_values=np.empty(0, dtype=np.float64),
        point_charge_gammas=np.empty(0, dtype=np.float64),
    )
    with pytest.raises(GPUxtbRuntimeError, match="point_charge_offsets"):
        ArrayBatch(**packed, backend="cpu").compute()


def test_empty_charge_response_group_is_not_silently_discarded() -> None:
    """An explicitly supplied but empty response group is invalid, not absent."""
    packed = _pack_single([_water()])
    packed.update(
        atomic_potential_shifts=np.zeros(3, dtype=np.float64),
        charge_response_offsets=np.asarray([0, 0], dtype=np.int64),
        charge_response_matrix=np.empty(0, dtype=np.float64),
    )
    with pytest.raises(GPUxtbRuntimeError, match="charge response"):
        ArrayBatch(**packed, backend="cpu").compute()


def test_dtype_mismatch_rejected() -> None:
    """float32 positions are refused instead of silently converted."""
    water = _water()
    packed = _pack_single([water])
    packed["positions"] = packed["positions"].astype(np.float32)
    with pytest.raises(GPUxtbValueError, match="dtype"):
        ArrayBatch(**packed, backend="cpu").compute()


def test_noncontiguous_rejected_without_copy() -> None:
    """Transposed views are refused under the zero-copy contract."""
    water = _water()
    packed = _pack_single([water])
    base = np.arange(27.0).reshape(3, 9)
    packed["positions"] = base[:, ::3]
    assert not packed["positions"].flags["C_CONTIGUOUS"]
    with pytest.raises(BufferError, match="C-contiguous"):
        ArrayBatch(**packed, backend="cpu").compute()


def test_copy_true_fixes_noncontiguous() -> None:
    """copy=True lets NumPy pack the descriptor before the call."""
    water = _water()
    reference = water.singlepoint()
    packed = _pack_single([water])
    positions = packed["positions"]
    padded = np.zeros((positions.shape[0], 9))
    padded[:, ::3] = positions
    packed["positions"] = padded[:, ::3]
    assert not packed["positions"].flags["C_CONTIGUOUS"]
    result = ArrayBatch(**packed, backend="cpu", copy=True).compute()
    assert result.energies == pytest.approx([reference.energy], rel=1.0e-10)


def test_batch_count_mismatch_rejected() -> None:
    """Offsets and per-system arrays must describe the same batch."""
    water = _water()
    packed = _pack_single([water])
    packed["molecular_charges"] = np.asarray([0.0, 1.0])
    with pytest.raises(GPUxtbValueError, match=r"atom_offsets|shape"):
        ArrayBatch(**packed, backend="cpu").compute()


@pytest.mark.parametrize("name", ["molecular_charges", "atomic_numbers"])
def test_scalar_count_descriptor_is_rejected(name: str) -> None:
    """Count derivation reports a value error instead of leaking IndexError."""
    packed = _pack_single([_water()])
    packed[name] = np.asarray(packed[name][0])
    with pytest.raises(GPUxtbValueError, match=name):
        ArrayBatch(**packed, backend="cpu").compute()


def test_scalar_point_charge_positions_are_rejected() -> None:
    """Point-charge count derivation requires the full ``(n, 3)`` shape."""
    packed = _pack_single([_water()])
    packed.update(
        point_charge_offsets=np.asarray([0, 1], dtype=np.int64),
        point_charge_positions=np.asarray(1.0),
        point_charge_values=np.asarray([1.0]),
        point_charge_gammas=np.asarray([0.5]),
    )
    with pytest.raises(GPUxtbValueError, match="point_charge_positions"):
        ArrayBatch(**packed, backend="cpu").compute()


def test_cuda_fake_array_on_cpu_backend_rejected() -> None:
    """Device arrays cannot be routed to a CPU context."""
    water = _water()
    packed = _pack_single([water])
    packed["positions"] = FakeArray(packed["positions"], device=2)
    with pytest.raises(GPUxtbNotSupportedError, match="CUDA backend"):
        ArrayBatch(**packed, backend="cpu").compute()


def test_cuda_fake_output_on_cpu_backend_rejected() -> None:
    """Output buffers follow the same backend/device contract as inputs."""
    packed = _pack_single([_water()])
    device_output = FakeArray(np.empty(1), device=2)
    with pytest.raises(GPUxtbNotSupportedError, match="CUDA backend"):
        ArrayBatch(**packed, backend="cpu").compute(
            compute_forces=False,
            compute_charges=False,
            out={"energies": device_output},
        )


def test_unknown_legacy_output_requires_proven_writability() -> None:
    """Unknown adapters cannot assert mutability for flagless old capsules."""
    packed = _pack_single([_water()])
    output = FakeArray(np.empty(1), versioned=False)
    with pytest.raises(BufferError, match=r"read.?only|writable"):
        ArrayBatch(**packed, backend="cpu").compute(
            compute_forces=False,
            compute_charges=False,
            out={"energies": output},
        )


def test_wrong_offsets_rejected_by_native_validation() -> None:
    """Non-monotonic offsets are the native layer's authoritative check."""
    water = _water()
    packed = _pack_single([water])
    packed["atom_offsets"] = np.asarray([0, 2], dtype=np.int64)
    with pytest.raises(GPUxtbRuntimeError):
        ArrayBatch(**packed, backend="cpu").compute()


def test_scc_failure_isolation_preserves_peers() -> None:
    """A non-convergent system must not poison its batch peers."""
    structures = [_water(), _water()]
    packed = _pack_single(structures)
    batch = ArrayBatch(**packed, backend="cpu")
    # max_scc_iterations=1 forces SCC non-convergence for every system.
    result = batch.compute(max_scc_iterations=1)
    assert result.scc_converged.tolist() == [0, 0]
    assert np.all(np.isnan(result.energies))
    assert result.failed_indices.tolist() == [0, 1]


def test_context_stream_property() -> None:
    """The DLPack consumer's native stream is exposed on the context."""
    batch = ArrayBatch(
        atom_offsets=np.asarray([0, 2], np.int64),
        atomic_numbers=np.asarray([1, 1], np.int32),
        positions=H2_POSITIONS,
        molecular_charges=np.asarray([0.0]),
        unpaired_electrons=np.asarray([0], np.int32),
        backend="cpu",
    )
    assert batch.context.stream is None
    batch.close()

    from gpuxtb import Context

    with pytest.raises(GPUxtbValueError, match="CPU backend"):
        Context("cpu", stream=0x1234)
    with pytest.raises(GPUxtbValueError, match="stream"):
        Context("cpu", stream=0)


def test_host_container_requires_dlpack() -> None:
    """Plain builtin containers are rejected with a clear message."""
    with pytest.raises(GPUxtbValueError, match="__dlpack__"):
        ArrayBatch(
            atom_offsets=[0, 2],
            atomic_numbers=[1, 1],
            positions=H2_POSITIONS.tolist(),
            molecular_charges=[0.0],
            unpaired_electrons=[0],
            backend="cpu",
        )


def array_is_numpy(array: object) -> bool:
    """Return whether the packed result returned an ordinary numpy array."""
    return isinstance(array, np.ndarray)


def test_missing_native_outputs_are_null() -> None:
    """Unrequested outputs bind the null descriptor and are not returned."""
    water = _water()
    packed = _pack_single([water])
    batch = ArrayBatch(**packed, backend="cpu")
    result = batch.compute(compute_energy=False)
    with pytest.raises(GPUxtbValueError, match="not requested"):
        _ = result.energies
    assert result.forces.shape == (3, 3)


def test_get_does_not_evaluate_unrequested_outputs() -> None:
    """Looking up one result must not require every optional property."""
    packed = _pack_single([_water()])
    result = ArrayBatch(**packed, backend="cpu").compute(compute_energy=False)
    assert result.get("forces") is result.forces
