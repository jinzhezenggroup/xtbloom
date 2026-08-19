"""Real-GPU GFN1 coverage for the packed Array API surface."""

from __future__ import annotations

import numpy as np
import pytest
from xtbloom import ArrayBatch, Calculator


@pytest.mark.cuda
def test_array_batch_gfn1_cuda_host_descriptors_match_cpu() -> None:
    arrays = {
        "atom_offsets": np.array([0, 2], dtype=np.int64),
        "atomic_numbers": np.array([1, 1], dtype=np.int32),
        "positions": np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=np.float64),
        "molecular_charges": np.array([0.0], dtype=np.float64),
        "unpaired_electrons": np.array([0], dtype=np.int32),
    }
    with ArrayBatch(**arrays, method="GFN1-xTB", backend="cuda") as batch:
        cuda_result = batch.compute()
    with Calculator(
        "GFN1-xTB",
        arrays["atomic_numbers"],
        arrays["positions"],
        backend="cpu",
    ) as calculator:
        cpu_result = calculator.singlepoint()

    np.testing.assert_allclose(
        cuda_result.energies, [cpu_result.energy], rtol=0.0, atol=5e-7
    )
    np.testing.assert_allclose(
        cuda_result.forces, cpu_result.forces, rtol=0.0, atol=5e-7
    )
    np.testing.assert_allclose(
        cuda_result.charges, cpu_result.charges, rtol=0.0, atol=1e-7
    )
