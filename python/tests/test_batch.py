"""Tests for the native ragged-batch interface (``gpuxtb.BatchCalculator``)."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import _cases
import numpy as np
import pytest
from gpuxtb import BatchCalculator, Calculator, Context, PointCharge, Structure, library
from gpuxtb.exceptions import GPUxtbRuntimeError, GPUxtbValueError

if TYPE_CHECKING:
    from collections.abc import Sequence

MOLECULAR_CASES = [
    "ketene",
    "nenacl",
    "h3_plus",
    "sif5_minus",
    "oh_radical",
]


def _make_structures(case_ids: Sequence[str], **kwargs: object) -> list[Structure]:
    """Build structures from named molecular conformance cases."""
    structures: list[Structure] = []
    for case_id in case_ids:
        case = _cases.case_by_id(case_id)
        numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
        structures.append(
            Structure(
                numbers,
                positions,
                charge=charge,
                uhf=uhf,
                spin_channels=spin,
                **kwargs,
            )
        )
    return structures


@pytest.mark.parametrize("case_id", MOLECULAR_CASES)
def test_batch_single_matches_serial(case_id: str) -> None:
    """Match a one-system batch to serial calculation output."""
    case = _cases.case_by_id(case_id)
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)

    single = Calculator(
        "GFN2-xTB", numbers, positions, charge=charge, uhf=uhf, spin_channels=spin
    )
    serial = single.singlepoint()

    batch = BatchCalculator(_make_structures([case_id]))
    result = batch.compute()
    assert result.energies[0] == pytest.approx(serial.energy, abs=1e-12)
    assert result.forces == pytest.approx(serial.forces, abs=1e-12)
    assert result[0].energy == pytest.approx(serial.energy, abs=1e-12)
    assert result.get("dipole_moments") is None


def test_batch_matches_goldens_all_molecular_cases() -> None:
    """Match every molecular batch slice to its conformance golden."""
    structures = _make_structures(MOLECULAR_CASES)
    batch = BatchCalculator(structures)
    result = batch.compute()
    tolerances = _cases.tolerances()
    for index, case_id in enumerate(MOLECULAR_CASES):
        golden = _cases.golden(_cases.case_by_id(case_id))
        item = result[index]
        assert item.energy == pytest.approx(
            golden["energy_hartree"], abs=tolerances["energy"]["atol"]
        ), case_id
        assert item.forces == pytest.approx(
            np.asarray(golden["forces_hartree_per_bohr"]).reshape(-1, 3),
            abs=tolerances["forces"]["atol"],
        ), case_id
    assert result[-1].energy == pytest.approx(result[len(result) - 1].energy, abs=0.0)
    assert result[-1].forces == pytest.approx(result[len(result) - 1].forces, abs=0.0)


def test_batch_preserves_healthy_peer_when_another_fails() -> None:
    """Preserve a successful peer when another system does not converge."""
    # h3+ converges in four iterations while NeNaCl does not.  The high-level
    # batch API must retain the healthy result and expose peer-local status.
    calculator = BatchCalculator(
        _make_structures(["h3_plus", "nenacl"]),
        backend="cpu",
        max_scc_iterations=4,
    )
    result = calculator.compute()
    assert result.failed_indices.tolist() == [1]
    assert result.per_system_status[0] == 0
    assert np.isfinite(result.energies[0])
    assert np.isnan(result.energies[1])
    with pytest.raises(GPUxtbRuntimeError):
        result.raise_for_status()
    with pytest.raises(GPUxtbRuntimeError):
        calculator.compute(raise_on_failure=True)


def test_batch_mixed_charge_and_spin_diagnostics() -> None:
    """Publish diagnostics for a batch with mixed charges and spin states."""
    charges = {
        "ketene": (0, 0),
        "h3_plus": (1, 0),
        "sif5_minus": (-1, 0),
        "oh_radical": (0, 1),
    }
    structures = []
    for case_id, (charge, uhf) in charges.items():
        numbers, positions, _, _, _ = _cases.structure_inputs(
            _cases.case_by_id(case_id)
        )
        structures.append(Structure(numbers, positions, charge=charge, uhf=uhf))
    batch = BatchCalculator(structures)
    result = batch.compute()
    assert len(result) == 4
    assert result.scc_converged.shape == (4,)
    assert (result.scc_converged == 1).all()
    assert result.per_system_status.shape == (4,)
    assert (result.per_system_status == 0).all()
    assert (result.scc_iterations > 0).all()


def test_batch_point_charges() -> None:
    """Match point-charge batch results to QM/MM conformance goldens."""
    case_ids = [
        "water_one_pc_gamma999",
        "water_dimer_6pc_hardness",
        "water_dimer_6pc_gamma999",
    ]
    structures = []
    for case_id in case_ids:
        case = _cases.case_by_id(case_id)
        numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
        point_positions, point_values, point_gammas = _cases.qmmm_points(case)
        structures.append(
            Structure(
                numbers,
                positions,
                charge=charge,
                uhf=uhf,
                spin_channels=spin,
                point_charges=PointCharge(point_positions, point_values, point_gammas),
            )
        )
    batch = BatchCalculator(structures)
    result = batch.compute()
    assert result.point_charge_forces is not None
    tolerances = _cases.tolerances()
    for index, case_id in enumerate(case_ids):
        golden = _cases.golden(_cases.case_by_id(case_id))
        item = result[index]
        assert item.energy == pytest.approx(
            golden["energy_hartree"], abs=tolerances["energy"]["atol"]
        ), case_id
        assert item.forces == pytest.approx(
            np.asarray(golden["forces_hartree_per_bohr"]).reshape(-1, 3),
            abs=tolerances["forces"]["atol"],
        ), case_id


def test_batch_empty_rejected() -> None:
    """Reject a batch that contains no structures."""
    from gpuxtb.exceptions import GPUxtbValueError

    with pytest.raises(GPUxtbValueError):
        BatchCalculator([])


def test_fixed_topology_plan_matches_compute_and_exposes_workspace() -> None:
    """A fixed-topology plan mirrors compute and exposes reusable workspace."""
    case = _cases.case_by_id("ketene")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)

    serial = Calculator(
        "GFN2-xTB", numbers, positions, charge=charge, uhf=uhf, spin_channels=spin
    )
    serial_result = serial.singlepoint()

    context = Context(backend="cpu")
    owners: list = []

    def host(
        values: Sequence[object],
        ctype: type[ctypes._SimpleCData],
        dtype: np.dtype,
    ) -> library.ConstBuffer:
        buf, owner = library.host_const(values, ctype, dtype)
        owners.append(owner)
        return buf

    atom_offsets = host([0, len(numbers)], ctypes.c_int64, np.int64)
    atomic_numbers = host(numbers, ctypes.c_int32, np.int32)
    positions_flat = host(np.asarray(positions).ravel(), ctypes.c_double, np.float64)
    molecular_charges = host([float(charge)], ctypes.c_double, np.float64)
    unpaired_electrons = host([int(uhf)], ctypes.c_int32, np.int32)
    spin_channels = host([int(spin)], ctypes.c_int32, np.int32)

    batch = library.Batch()
    context._create()
    library._check_init(
        "gpuxtb_batch_init",
        library.load_library().gpuxtb_batch_init(
            ctypes.byref(batch), ctypes.sizeof(batch)
        ),
    )
    batch.batch_size = 1
    batch.total_atoms = len(numbers)
    batch.atom_offsets = atom_offsets
    batch.atomic_numbers = atomic_numbers
    batch.positions = positions_flat
    batch.molecular_charges = molecular_charges
    batch.unpaired_electrons = unpaired_electrons
    batch.spin_channels = spin_channels

    full_flags = (
        library.COMPUTE_ENERGY | library.COMPUTE_FORCES | library.COMPUTE_ATOMIC_CHARGES
    )
    options = library.ComputeOptions()
    library._check_init(
        "gpuxtb_compute_options_init",
        library.load_library().gpuxtb_compute_options_init(
            ctypes.byref(options), ctypes.sizeof(options)
        ),
    )
    options.flags = full_flags
    plan = context.create_plan(batch, options)
    assert plan.handle

    workspace = plan.query_workspace(full_flags)
    assert workspace.host_required_bytes > 0
    assert workspace.host_required_alignment >= 8
    assert workspace.device_required_bytes == 0  # CPU backend has no device memory

    energy_options = library.ComputeOptions()
    library._check_init(
        "gpuxtb_compute_options_init",
        library.load_library().gpuxtb_compute_options_init(
            ctypes.byref(energy_options), ctypes.sizeof(energy_options)
        ),
    )
    energy_options.flags = library.COMPUTE_ENERGY
    energy_plan = context.create_plan(batch, energy_options)
    energy_only = energy_plan.query_workspace(library.COMPUTE_ENERGY)
    assert workspace.host_required_bytes >= energy_only.host_required_bytes
    result = library.BatchResult()
    library._check_init(
        "gpuxtb_batch_result_init",
        library.load_library().gpuxtb_batch_result_init(
            ctypes.byref(result), ctypes.sizeof(result)
        ),
    )
    energies, energy_owner = library.empty_result_shape(1, ctypes.c_double, np.float64)
    forces, forces_owner = library.empty_result_shape(
        3 * len(numbers), ctypes.c_double, np.float64
    )
    charges, charges_owner = library.empty_result_shape(
        len(numbers), ctypes.c_double, np.float64
    )
    iterations, iterations_owner = library.empty_result_shape(
        1, ctypes.c_int32, np.int32
    )
    converged, converged_owner = library.empty_result_shape(1, ctypes.c_uint8, np.uint8)
    statuses, statuses_owner = library.empty_result_shape(1, ctypes.c_int32, np.int32)
    owners.extend(
        [
            energy_owner,
            forces_owner,
            charges_owner,
            iterations_owner,
            converged_owner,
            statuses_owner,
        ]
    )
    result.energies = energies
    result.forces = forces
    result.atomic_charges = charges
    result.scc_iterations = iterations
    result.scc_converged = converged
    result.per_system_status = statuses

    plan.compute(batch, options, result)
    assert statuses_owner[0] == library.STATUS_SUCCESS
    assert converged_owner[0] == 1
    assert iterations_owner[0] > 0
    assert energy_owner[0] == pytest.approx(serial_result.energy, abs=1e-12)
    assert np.allclose(forces_owner.reshape(-1, 3), serial_result.forces, atol=1e-12)

    # Reuse the same fixed topology with a changed geometry.
    shifted = np.asarray(positions, dtype=np.float64).copy()
    shifted[0, 0] += 0.01
    positions_flat2 = host(shifted.ravel(), ctypes.c_double, np.float64)
    batch.positions = positions_flat2
    energy2, energy2_owner = library.empty_result_shape(1, ctypes.c_double, np.float64)
    owners.append(energy2_owner)
    result.energies = energy2
    plan.compute(batch, options, result)
    assert statuses_owner[0] == library.STATUS_SUCCESS
    assert np.isfinite(energy2_owner[0])

    plan.destroy()
    with pytest.raises(GPUxtbRuntimeError, match="plan is closed"):
        plan.query_workspace(full_flags)
    context.close()
    with pytest.raises(GPUxtbRuntimeError, match="plan is closed"):
        energy_plan.query_workspace(library.COMPUTE_ENERGY)


def test_batch_warm_start_reuses_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """``BatchCalculator(warm_start=True)`` reconverges from the previous batch."""
    modes: list[int] = []
    original = library.compute_checked

    def recording(
        context: object, batch: object, options: object, result: object
    ) -> None:
        modes.append(int(options.scc_start_mode))  # type: ignore[attr-defined]
        return original(context, batch, options, result)  # type: ignore[arg-type]

    monkeypatch.setattr(library, "compute_checked", recording)

    structures = _make_structures(["ketene"])
    batch = BatchCalculator(structures, warm_start=True)
    energies: list[float] = []
    for step in range(3):
        positions = np.asarray(structures[0].positions, dtype=np.float64)
        positions[step % len(positions), 0] += 0.02
        structures[0].update(positions=positions)
        result = batch.compute()
        result.raise_for_status()
        energies.append(float(result.energies[0]))
    assert all(np.isfinite(energies))
    assert modes == [
        library.SCC_START_FRESH,
        *([library.SCC_START_WARM] * 2),
    ]


def test_batch_warm_start_rejects_auto_slicing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not reuse one chunk's checkpoint as another chunk's initial state."""
    calls = 0

    def unexpected_compute(*args: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("native compute must not run for this invalid combination")

    monkeypatch.setattr(library, "compute_checked", unexpected_compute)
    structures = _make_structures(["ketene", "ketene"])
    batch = BatchCalculator(structures, warm_start=True)

    with pytest.raises(GPUxtbValueError, match="cannot be combined"):
        batch.compute(auto_batch_size=1)

    assert calls == 0


@pytest.mark.parametrize("auto_batch_size", [None, 1])
def test_batch_per_system_efield_matches_serial(
    auto_batch_size: int | None,
) -> None:
    """Each batch member attaches its own uniform electric field."""
    angstrom_per_bohr = 1.8897261246257702
    xyz = [
        [0.0, 0.0, -0.2358784530],
        [0.0, 1.4270063049, 1.0081495306],
        [0.0, -1.4270063049, 1.0081495306],
    ]
    plain = Structure(
        np.array([8, 1, 1]),
        np.array([[c * angstrom_per_bohr for c in atom] for atom in xyz]),
    )
    fielded = Structure(
        np.array([8, 1, 1]),
        np.array([[c * angstrom_per_bohr for c in atom] for atom in xyz]),
        efield=[0.001, 0.002, -0.0015],
    )
    batch = BatchCalculator([plain, fielded, plain])
    # Dipoles are a logical-batch property: when auto-slicing separates the
    # fielded system from plain peers, every chunk still requests the same
    # result shape so merging retains one (nsystems, 3) array.
    result = batch.compute(auto_batch_size=auto_batch_size)

    serial_field = Calculator(
        "GFN2-xTB",
        np.array([8, 1, 1]),
        np.array([[c * angstrom_per_bohr for c in atom] for atom in xyz]),
        efield=[0.001, 0.002, -0.0015],
    ).singlepoint()
    assert result[1].energy == pytest.approx(serial_field.energy, abs=1e-10)
    assert result[1].forces == pytest.approx(serial_field.forces, abs=1e-10)
    assert result.dipole_moments is not None
    assert result.dipole_moments.shape == (3, 3)
    assert result[1].dipole_moments == pytest.approx(
        serial_field.dipole_moments, abs=1e-10
    )
    assert result.get("dipole_moments") is result.dipole_moments
    assert result[0].energy == pytest.approx(result[2].energy, abs=1e-12)
