"""GFN1 coverage for the public packed Array API/DLPack surface."""

from __future__ import annotations

import numpy as np
import pytest
from xtbloom import ArrayBatch, Calculator, compute_arrays
from xtbloom.exceptions import XTBloomValueError


def _h2_arrays() -> dict[str, np.ndarray]:
    return {
        "atom_offsets": np.array([0, 2], dtype=np.int64),
        "atomic_numbers": np.array([1, 1], dtype=np.int32),
        "positions": np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=np.float64),
        "molecular_charges": np.array([0.0], dtype=np.float64),
        "unpaired_electrons": np.array([0], dtype=np.int32),
    }


def test_array_batch_rejects_unknown_method_before_native_execution() -> None:
    arrays = _h2_arrays()
    with pytest.raises(XTBloomValueError):
        ArrayBatch(**arrays, method="GFN0-xTB", backend="cpu")


@pytest.mark.parametrize("method", ["GFN1-xTB", "GFN1"])
def test_array_batch_gfn1_cpu_matches_calculator(method: str) -> None:
    arrays = _h2_arrays()
    with ArrayBatch(**arrays, method=method, backend="cpu") as batch:
        packed = batch.compute()
    with Calculator(
        "GFN1-xTB",
        arrays["atomic_numbers"],
        arrays["positions"],
        backend="cpu",
    ) as calculator:
        reference = calculator.singlepoint()

    np.testing.assert_allclose(
        packed.energies, [reference.energy], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(packed.forces, reference.forces, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(packed.charges, reference.charges, rtol=0.0, atol=1e-12)


def test_compute_arrays_forwards_gfn1_method() -> None:
    arrays = _h2_arrays()
    direct = compute_arrays(**arrays, method="GFN1-xTB", backend="cpu")
    with ArrayBatch(**arrays, method="GFN1-xTB", backend="cpu") as batch:
        explicit = batch.compute()

    np.testing.assert_allclose(direct.energies, explicit.energies, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(direct.forces, explicit.forces, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(direct.charges, explicit.charges, rtol=0.0, atol=0.0)


def test_array_batch_default_remains_gfn2() -> None:
    arrays = _h2_arrays()
    with ArrayBatch(**arrays, backend="cpu") as batch:
        assert batch.method == "GFN2-xTB"
        default = batch.compute()
    with Calculator(
        "GFN2-xTB",
        arrays["atomic_numbers"],
        arrays["positions"],
        backend="cpu",
    ) as calculator:
        reference = calculator.singlepoint()
    np.testing.assert_allclose(
        default.energies, [reference.energy], rtol=0.0, atol=1e-12
    )
