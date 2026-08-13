"""Regression tests for the installed-Pyodide-wheel inference harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import xtbloom
from xtbloom import interface, library

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import TracebackType

    import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPOSITORY / "python" / "ci" / "run-pyodide-wheel-test.py"
SPEC = importlib.util.spec_from_file_location(
    "xtbloom_pyodide_wheel_test", HARNESS_PATH
)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def test_complete_invariants_request_field_free_dipoles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request dipoles directly without manufacturing a zero-field attachment."""
    geometries = [
        SimpleNamespace(
            case_id="plain",
            atomic_numbers=[1],
            positions=[0.0, 0.0, 0.0],
            molecular_charge=0,
            unpaired_electrons=0,
            spin_channels=1,
            point_positions=[],
            point_values=[],
            point_gammas=[],
            efield=None,
        ),
        SimpleNamespace(
            case_id="field_and_point",
            atomic_numbers=[8],
            positions=[1.0, 2.0, 3.0],
            molecular_charge=-1,
            unpaired_electrons=0,
            spin_channels=1,
            point_positions=[4.0, 5.0, 6.0],
            point_values=[0.25],
            point_gammas=[0.1],
            efield=[0.003, -0.004, 0.005],
        ),
    ]
    captured_structures: list[xtbloom.Structure] = []
    captured_kwargs: dict[str, object] = {}

    class FakeContext:
        def __init__(self, backend: str) -> None:
            assert backend == "cpu"

        def __enter__(self) -> FakeContext:  # noqa: PYI034 - compact test double
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            return None

    def fake_compute_batch(
        _context: object,
        structures: Sequence[xtbloom.Structure],
        **kwargs: object,
    ) -> SimpleNamespace:
        captured_structures.extend(structures)
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            energies=np.array([1.0, 2.0]),
            forces=np.arange(6.0).reshape(2, 3),
            charges=np.array([0.1, -1.1]),
            point_charge_forces=np.array([[7.0, 8.0, 9.0]]),
            dipole_moments=np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]),
            per_system_status=np.array(
                [library.STATUS_SUCCESS, library.STATUS_SUCCESS]
            ),
            scc_converged=np.array([1, 1]),
            atom_offsets=np.array([0, 1, 2]),
            point_offsets=np.array([0, 0, 1]),
        )

    def run_invariant_checks(
        solver: Callable[[Sequence[SimpleNamespace]], list[SimpleNamespace]],
        supplied_geometries: Sequence[SimpleNamespace],
        homogeneous_case_ids: Sequence[str],
    ) -> list[str]:
        assert supplied_geometries == geometries
        assert homogeneous_case_ids == ()
        results = solver(supplied_geometries)
        assert results[0].dipoles == [10.0, 11.0, 12.0]
        assert results[0].efield is None
        assert results[0].point_forces == []
        assert results[1].dipoles == [13.0, 14.0, 15.0]
        assert results[1].efield == [0.003, -0.004, 0.005]
        assert results[1].point_forces == [7.0, 8.0, 9.0]
        return []

    fake_invariants = SimpleNamespace(
        conformance=SimpleNamespace(load_json=lambda _path: {}),
        public_api=SimpleNamespace(supported_cases=lambda *_args: []),
        load_geometries=lambda *_args: geometries,
        select_homogeneous_case_ids=lambda _geometries: (),
        InvariantResult=SimpleNamespace,
        run_invariant_checks=run_invariant_checks,
    )
    monkeypatch.setattr(HARNESS, "_load_invariants", lambda _root: fake_invariants)
    monkeypatch.setattr(xtbloom, "Context", FakeContext)
    monkeypatch.setattr(interface, "_compute_batch", fake_compute_batch)

    HARNESS._run_complete_invariants(REPOSITORY)

    assert captured_structures[0].efield is None
    np.testing.assert_allclose(captured_structures[1].efield, geometries[1].efield)
    flags = captured_kwargs["flags"]
    assert flags == (
        library.COMPUTE_ENERGY
        | library.COMPUTE_FORCES
        | library.COMPUTE_ATOMIC_CHARGES
        | library.COMPUTE_POINT_CHARGE_FORCES
        | library.COMPUTE_DIPOLE_MOMENTS
    )
