"""Integration smoke test for ASE-driven molecular dynamics."""

from __future__ import annotations

import numpy as np
import pytest

ase = pytest.importorskip("ase")
import xtbloom.library as _library  # noqa: E402
from ase import Atoms, units  # noqa: E402
from ase.md.verlet import VelocityVerlet  # noqa: E402
from xtbloom.ase import XTBloom  # noqa: E402


def test_velocity_verlet_drives_repeated_xtbloom_force_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a short real ASE trajectory through the xTBloom calculator."""
    calls: list[int] = []
    original = _library.compute_checked

    def recording(
        context: object, batch: object, options: object, result: object
    ) -> None:
        calls.append(int(options.scc_start_mode))  # type: ignore[attr-defined]
        return original(context, batch, options, result)  # type: ignore[arg-type]

    monkeypatch.setattr(_library, "compute_checked", recording)
    atoms = Atoms(
        symbols="OH2",
        positions=[
            [0.0000, 0.0000, 0.0000],
            [0.7586, 0.0000, 0.5043],
            [-0.7586, 0.0000, 0.5043],
        ],
        pbc=False,
    )
    atoms.set_velocities(np.zeros((3, 3)))
    calculator = XTBloom(method="GFN2-xTB", backend="cpu", warm_start=True)
    atoms.calc = calculator
    try:
        dynamics = VelocityVerlet(atoms, timestep=0.1 * units.fs)
        dynamics.run(2)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
    finally:
        calculator.close()

    assert len(calls) >= 3
    assert calls[0] == _library.SCC_START_FRESH
    assert all(mode == _library.SCC_START_WARM for mode in calls[1:])
    assert np.isfinite(energy)
    assert np.isfinite(forces).all()
    assert np.isfinite(atoms.positions).all()
