"""Tests for automatic batch slicing (``BatchCalculator.compute(auto_batch_size=...)``).

Auto slicing must be exact: every system stays whole, per-system results (and
point-charge/force slices) must be bit-identical to an unsliced run, and the
machine-adaptive CUDA budget must degrade gracefully on hosts without CUDA.
"""

from __future__ import annotations

import _cases
import pytest
from gpuxtb import BatchCalculator, Context, PointCharge, Structure, library
from gpuxtb.exceptions import GPUxtbRuntimeError, GPUxtbValueError
from gpuxtb.interface import _slice_by_total_atoms


def _make_structures(case_ids, repeats=1, **kwargs):
    structures = []
    for _ in range(repeats):
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


def _library_has_cuda() -> bool:
    try:
        with Context("cuda"):
            pass
        return True
    except GPUxtbRuntimeError:
        return False


def test_slice_keeps_systems_whole_and_bounded():
    case = _cases.case_by_id("ketene")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    # ketene has 5 atoms; a budget of 7 atoms must group 5+... first chunk is
    # full, second is one system, and a budget of 2 still keeps the 5-atom
    # system in its own oversized chunk instead of dropping it.
    structures = [Structure(numbers, positions) for _ in range(3)]
    chunks = _slice_by_total_atoms(structures, max_total_atoms=7)
    assert [len(chunk) for chunk in chunks] == [1, 1, 1]
    assert sum(len(chunk) for chunk in chunks) == 3

    chunks = _slice_by_total_atoms(structures, max_total_atoms=2)
    assert len(chunks) == 3
    assert all(len(chunk) == 1 for chunk in chunks)

    with pytest.raises(GPUxtbValueError):
        _slice_by_total_atoms(structures, max_total_atoms=0)


def test_auto_batch_int_matches_unsliced():
    structures = _make_structures(
        ["ketene", "h3_plus", "oh_radical", "sif5_minus"], repeats=3
    )
    unsliced = BatchCalculator(structures, backend="cpu").compute()
    sliced = BatchCalculator(structures, backend="cpu").compute(auto_batch_size=6)
    assert len(sliced) == len(unsliced)
    assert sliced.energies == pytest.approx(unsliced.energies, abs=0.0)
    assert sliced.forces == pytest.approx(unsliced.forces, abs=0.0)
    assert sliced.charges == pytest.approx(unsliced.charges, abs=0.0)
    assert (sliced.scc_converged == unsliced.scc_converged).all()
    assert (sliced.per_system_status == unsliced.per_system_status).all()
    for index in range(len(unsliced)):
        assert sliced[index].energy == pytest.approx(unsliced[index].energy, abs=0.0)
        assert sliced[index].forces == pytest.approx(unsliced[index].forces, abs=0.0)


def test_auto_batch_point_charges_matches_unsliced():
    case_ids = [
        "water_one_pc_gamma999",
        "water_dimer_6pc_hardness",
        "water_dimer_6pc_gamma999",
    ]
    structures = []
    for _ in range(2):
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
                    point_charges=PointCharge(
                        point_positions, point_values, point_gammas
                    ),
                )
            )
    unsliced = BatchCalculator(structures, backend="cpu").compute()
    sliced = BatchCalculator(structures, backend="cpu").compute(auto_batch_size=4)
    assert sliced.point_charge_forces is not None
    assert sliced.point_charge_forces == pytest.approx(
        unsliced.point_charge_forces, abs=0.0
    )
    assert sliced.energies == pytest.approx(unsliced.energies, abs=0.0)
    for index in range(len(unsliced)):
        assert sliced[index].point_charge_forces == pytest.approx(
            unsliced[index].point_charge_forces, abs=0.0
        )


def test_auto_batch_true_defaults_to_single_call_on_small_batch():
    # A batch smaller than the calibration sample must run unchanged (single
    # call) yet still produce a valid result via the auto path.
    structures = _make_structures(["ketene"])
    unsliced = BatchCalculator(structures).compute()
    auto = BatchCalculator(structures).compute(auto_batch_size=True)
    assert auto.energies == pytest.approx(unsliced.energies, abs=1e-12)
    assert auto.forces == pytest.approx(unsliced.forces, abs=1e-12)


def test_auto_batch_rejects_invalid_limit():
    structures = _make_structures(["ketene"])
    with pytest.raises(GPUxtbValueError):
        BatchCalculator(structures).compute(auto_batch_size=0)
    with pytest.raises(GPUxtbValueError):
        BatchCalculator(structures).compute(auto_batch_size=-3)


@pytest.mark.cuda
def test_auto_batch_cuda_matches_unsliced():
    if not _library_has_cuda():
        pytest.skip("CUDA backend is not available on this host")
    structures = _make_structures(
        ["ketene", "h3_plus", "water_dimer_6pc_hardness"], repeats=2
    )
    unsliced = BatchCalculator(structures, backend="cuda").compute()
    sliced = BatchCalculator(structures, backend="cuda").compute(auto_batch_size=4)
    assert sliced.energies == pytest.approx(unsliced.energies, abs=1e-12)
    assert sliced.forces == pytest.approx(unsliced.forces, abs=1e-12)

    # The machine-adaptive path must also stay accurate on a real device.
    auto = BatchCalculator(structures, backend="cuda").compute(auto_batch_size=True)
    assert auto.energies == pytest.approx(unsliced.energies, abs=1e-12)
    assert auto.forces == pytest.approx(unsliced.forces, abs=1e-12)


def test_device_memory_info_optional_and_typed():
    info = library.device_memory_info(0)
    if info is not None:
        free_bytes, total_bytes = info
        assert total_bytes > 0
        assert 0 <= free_bytes <= total_bytes
