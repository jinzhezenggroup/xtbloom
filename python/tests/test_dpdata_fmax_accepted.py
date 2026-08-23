"""Focused regression for accepted-geometry fmax convergence."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("dpdata")
from xtbloom.dpdata import _FORCE_TO_EV_ANG, XTBloomMinimizer


def test_accepted_geometry_uses_per_atom_force_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep an accepted frame active when only its force components pass fmax."""
    component = 0.004 / _FORCE_TO_EV_ANG
    calls: list[np.ndarray] = []

    class FakeBatchCalculator:
        def __init__(
            self, structures: list[object], _method: str, **_kwargs: object
        ) -> None:
            self._structures = structures

        def compute(self) -> list[SimpleNamespace]:
            call = len(calls)
            position = self._structures[0].positions.copy()  # type: ignore[attr-defined]
            calls.append(position)
            if call == 0:
                energy = 0.0
                force = np.ones_like(position)
            elif call == 1:
                energy = -1.0
                force = np.full_like(position, component)
            else:
                energy = -2.0
                force = np.zeros_like(position)
            return [SimpleNamespace(energy=energy, forces=force)]

        def close(self) -> None:
            pass

    monkeypatch.setattr("xtbloom.dpdata.BatchCalculator", FakeBatchCalculator)
    data = {
        "atom_names": ["H"],
        "atom_types": np.array([0], dtype=np.int64),
        "coords": np.zeros((1, 1, 3), dtype=np.float64),
        "nopbc": True,
    }

    XTBloomMinimizer(fmax=0.005, max_steps=2).minimize(data)

    # Call 1 is an energy-accepted geometry whose individual components pass
    # fmax while sqrt(3) * 0.004 eV/Å exceeds it. The fixed norm criterion must
    # therefore schedule call 2 instead of declaring convergence at call 1.
    assert len(calls) == 3
