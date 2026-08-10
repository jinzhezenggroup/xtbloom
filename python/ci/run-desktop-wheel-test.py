#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise an installed desktop wheel and its private eigensolver provider."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xtbloom
from xtbloom import Calculator

NUMBERS = np.array([8, 1, 1], dtype=np.int32)
POSITIONS = np.array(
    [
        [0.0, 0.0, -0.73578586109551],
        [1.44183152868459, 0.0, 0.36789293054775],
        [-1.44183152868459, 0.0, 0.36789293054775],
    ],
    dtype=np.float64,
)


def _singlepoint() -> tuple[float, np.ndarray]:
    calculator = Calculator("GFN2-xTB", NUMBERS, POSITIONS, backend="cpu")
    result = calculator.singlepoint()
    if not result.scc_converged:
        raise RuntimeError("installed desktop wheel did not converge water")
    if not np.isfinite(result.energy) or not np.isfinite(result.forces).all():
        raise RuntimeError("installed desktop wheel returned non-finite results")
    return float(result.energy), np.asarray(result.forces)


def _private_provider() -> Path:
    package = Path(xtbloom.__file__).resolve().parent
    candidates = [
        *sorted((package / "lib").glob("libxtbloom_blas-*.dylib")),
        *sorted((package / "bin").glob("xtbloom_openblas-*.dll")),
    ]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise RuntimeError(
            "installed desktop wheel must contain one private provider; "
            f"found {present}"
        )
    return present[0]


def _prove_missing_provider_fails(provider: Path) -> None:
    disabled = provider.with_name(provider.name + ".disabled")
    provider.rename(disabled)
    try:
        code = """
import numpy as np
from xtbloom import Calculator

calculator = Calculator(
    "GFN2-xTB",
    np.array([8, 1, 1], dtype=np.int32),
    np.array(
        [
            [0.0, 0.0, -0.73578586109551],
            [1.44183152868459, 0.0, 0.36789293054775],
            [-1.44183152868459, 0.0, 0.36789293054775],
        ],
        dtype=np.float64,
    ),
    backend="cpu",
)
try:
    calculator.singlepoint()
except Exception as error:
    message = str(error).lower()
    if (
        "openblas" not in message
        and "linear-algebra" not in message
        and "backend" not in message
    ):
        raise
else:
    raise SystemExit(
        "CPU inference unexpectedly succeeded without the private provider"
    )
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "missing-provider child did not fail through the expected "
                "backend path:\n" + completed.stdout + completed.stderr
            )
    finally:
        disabled.rename(provider)


def main() -> int:
    """Run provider, host-coexistence, correctness, and concurrency checks."""
    try:
        importlib.metadata.distribution("scipy-openblas32")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("scipy-openblas32 leaked into the wheel test environment")

    provider = _private_provider()
    _prove_missing_provider_fails(provider)

    # Exercise the host array backend before and after xTBloom initializes its
    # separately verified private provider. This is a numerical coexistence
    # check; the runtime factory independently proves that every setter and
    # dispatch symbol belongs to the exact private image it opened.
    host_matrix = np.array([[2.0, 0.5], [0.5, 1.0]])
    host_before = np.linalg.eigvalsh(host_matrix)
    reference_energy, reference_forces = _singlepoint()
    host_after = np.linalg.eigvalsh(host_matrix)
    np.testing.assert_allclose(host_after, host_before, rtol=0.0, atol=0.0)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: _singlepoint(), range(8)))
    for energy, forces in results:
        np.testing.assert_allclose(energy, reference_energy, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(forces, reference_forces, rtol=0.0, atol=1.0e-12)

    print(  # noqa: T201 - CI validation report
        f"xTBloom desktop wheel CPU inference passed: energy={reference_energy:.16g}; "
        f"provider={provider.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
